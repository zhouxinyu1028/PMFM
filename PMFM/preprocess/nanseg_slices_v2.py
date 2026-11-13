import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess.news_numpy_output import (body_dice, decode_mr, index_label_files,
                                          otsu_threshold, read_labels, register_mr_to_ct)



def ncc(a, b, mask):

    a = a[mask].astype(np.float64)
    b = b[mask].astype(np.float64)
    if a.size < 100 or a.std() < 1e-8 or b.std() < 1e-8:
        return float('nan')
    return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))


def register_deformable(ct, mr, rigid, work_mm=2.0, ctrl_mm=30.0, sampling=0.10):

    def shrink_to(im, mm):
        sp = im.GetSpacing()
        return sitk.Cast(sitk.Shrink(im, [max(1, int(round(mm / s))) for s in sp]), sitk.sitkFloat32)

    ct_c, mr_c = shrink_to(ct, work_mm), shrink_to(mr, work_mm)
    mesh = [max(1, int(round((sz - 1) * sp / ctrl_mm)))
            for sz, sp in zip(ct_c.GetSize(), ct_c.GetSpacing())]
    bspline = sitk.BSplineTransformInitializer(ct_c, mesh, order=3)

    composite = sitk.CompositeTransform(rigid)   # 后加入的 bspline 先作用
    composite.AddTransform(bspline)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling, seed=42)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100,
                             maximumNumberOfCorrections=5)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(composite, inPlace=False)

    t0 = time.time()
    transform = reg.Execute(ct_c, mr_c)
    return transform, time.time() - t0, mesh


def max_displacement(transform, image, steps=6):

    size = image.GetSize()
    disp = []
    for z in range(0, size[2], max(1, size[2] // steps)):
        for y in range(0, size[1], max(1, size[1] // steps)):
            for x in range(0, size[0], max(1, size[0] // steps)):
                p = np.asarray(image.TransformIndexToPhysicalPoint((x, y, z)), dtype=np.float64)
                q = np.asarray(transform.TransformPoint(tuple(p)), dtype=np.float64)
                disp.append(np.linalg.norm(q - p))
    disp = np.asarray(disp)
    return float(disp.max()), float(disp.mean())



def count_organ_slices(raw_root, label_names, cases):
    per_organ = Counter()
    per_case = defaultdict(dict)      # case -> label -> [z...]
    for case_dir in cases:
        for label_id, seg_path in index_label_files(case_dir, label_names):
            if seg_path is None:
                continue

            seg_img = sitk.ReadImage(str(seg_path))
            arr = sitk.GetArrayFromImage(seg_img)
            zs = np.flatnonzero(arr.reshape(arr.shape[0], -1).any(axis=1))
            per_case[case_dir.name][label_id] = zs.tolist()
            per_organ[label_id] += len(zs)
    return per_organ, per_case



class HaNSegWriterV2:
    def __init__(self, raw_root, out_root, label_names, size=512, registration='rigid',
                 max_slices=20, protect_below=400, min_rel_area=0.02, min_area_px=4,
                 drop_mr_empty=True, val_ratio=0.15, test_ratio=0.15, seed=42,
                 max_deform_mm=25.0, min_gain=0.0, dry_run=False):
        self.raw_root = Path(raw_root)
        self.out_root = Path(out_root)
        self.label_names = label_names
        self.size = size
        self.registration = registration
        self.max_slices = max_slices
        self.protect_below = protect_below
        self.min_rel_area = min_rel_area
        self.min_area_px = min_area_px
        self.drop_mr_empty = drop_mr_empty
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.max_deform_mm = max_deform_mm
        self.min_gain = min_gain
        self.dry_run = dry_run

        self.slices_dir = self.out_root / 'slices'
        self.samples_dir = self.out_root / 'samples'
        if not dry_run:
            self.slices_dir.mkdir(parents=True, exist_ok=True)
            for label_id in range(len(label_names)):
                (self.samples_dir / str(label_id)).mkdir(parents=True, exist_ok=True)


    def prepare_case(self, case_dir):
        ct = sitk.ReadImage(str(next(case_dir.glob('*_IMG_CT.nrrd'))))
        mr = decode_mr(sitk.ReadImage(str(next(case_dir.glob('*_IMG_MR*.nrrd')))))


        target_size = (self.size, self.size, ct.GetSize()[2])
        sp = np.asarray(ct.GetSpacing(), dtype=np.float64)
        tsp = np.array([sp[0] * ct.GetSize()[0] / self.size,
                        sp[1] * ct.GetSize()[1] / self.size,
                        sp[2]])
        direction = np.asarray(ct.GetDirection(), dtype=np.float64).reshape(3, 3)
        target_origin = np.asarray(ct.GetOrigin(), dtype=np.float64) + direction @ (0.5 * (tsp - sp))

        ct_grid = sitk.Resample(ct, target_size, sitk.Transform(), sitk.sitkLinear,
                                tuple(target_origin), tuple(tsp), ct.GetDirection(),
                                0.0, sitk.sitkFloat32)
        ct_np = sitk.GetArrayFromImage(ct_grid).astype(np.float32)


        mr_raw = sitk.GetArrayFromImage(
            sitk.Resample(mr, ct_grid, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32))
        dice_rigid_init = body_dice(ct_np, mr_raw)

        best = None
        for init_name, init in (('GEOMETRY', sitk.CenteredTransformInitializerFilter.GEOMETRY),
                                ('MOMENTS', sitk.CenteredTransformInitializerFilter.MOMENTS)):
            try:
                transform, metric, stop = register_mr_to_ct(ct, mr, init)
            except RuntimeError as exc:
                print(f'    刚性配准({init_name})失败: {exc}', flush=True)
                continue
            cand = sitk.GetArrayFromImage(
                sitk.Resample(sitk.Cast(mr, sitk.sitkFloat32), ct_grid, transform, sitk.sitkLinear,
                              0.0, sitk.sitkFloat32))
            dice = body_dice(ct_np, cand)
            if best is None or dice > best['dice']:
                best = dict(dice=dice, transform=transform, metric=metric, stop=stop,
                            init=init_name, mr=cand)
        if best is None:
            best = dict(dice=dice_rigid_init, transform=sitk.Transform(3, sitk.sitkIdentity),
                        metric=float('nan'), stop='failed', init='none', mr=mr_raw)

        qc = dict(dice_before=dice_rigid_init, dice_rigid=best['dice'], dice_final=best['dice'],
                  init=best['init'], metric=best['metric'], stop=best['stop'],
                  deform='none', deform_time=0.0, max_disp=0.0, mean_disp=0.0, mesh='-')
        mr_final = best['mr']
        transform = best['transform']


        if self.registration == 'deformable' and not self.dry_run:
            try:
                deform, secs, mesh = register_deformable(ct, mr, transform)
                cand = sitk.GetArrayFromImage(
                    sitk.Resample(sitk.Cast(mr, sitk.sitkFloat32), ct_grid, deform, sitk.sitkLinear,
                                  0.0, sitk.sitkFloat32))
                dice_def = body_dice(ct_np, cand)
                mx, mean = max_displacement(deform, ct_grid)
                qc.update(deform_time=secs, max_disp=mx, mean_disp=mean, mesh=str(mesh))
                ok = (dice_def >= best['dice'] + self.min_gain) and (mx <= self.max_deform_mm)
                qc['deform'] = 'accepted' if ok else f'rejected(dice {best["dice"]:.3f}->{dice_def:.3f}, max_disp {mx:.1f}mm)'
                if ok:
                    mr_final, transform, qc['dice_final'] = cand, deform, dice_def
            except Exception as exc:                      # noqa: BLE001 配准失败不应中断整批
                qc['deform'] = f'failed({type(exc).__name__}: {exc})'


        body = mr_final.reshape(mr_final.shape[0], -1).max(axis=1) > 0
        if body.any():
            a, b = ct_np[body], mr_final[body]
            thr = otsu_threshold(b[b > 0]) if (b > 0).any() else 0.0
            m = (a > -300) & (b > thr)
            qc['ncc'] = ncc(a, b, m)
        else:
            qc['ncc'] = float('nan')

        return ct_grid, ct_np, mr_final, transform, qc


    def select_slices(self, per_organ_areas, mr_signal, protected):
        """
            per_organ_areas: {label_id: [(z, area), ...]}
            mr_signal: {z: bool}
            返回 (kept: {label: [z...]}, stats: dict)
        """
        kept, stats = {}, defaultdict(int)
        for label_id, items in per_organ_areas.items():
            items = sorted(items)
            if not items:
                continue
            mx = max(a for _, a in items)

            filt = [(z, a) for z, a in items if a >= max(self.min_area_px, self.min_rel_area * mx)]
            stats['sliver_dropped'] += len(items) - len(filt)
            if not filt:
                filt = [max(items, key=lambda t: t[1])]  

            is_small = protected.get(label_id, False)
  
            if self.drop_mr_empty and not is_small:
                signal = [t for t in filt if mr_signal.get(t[0], False)]
                if len(signal) >= max(2, int(np.ceil(0.3 * len(filt)))):
                    stats['mr_empty_dropped'] += len(filt) - len(signal)
                    filt = signal
                else:
                    stats['mr_empty_kept'] += len(filt) - len(signal)


            if not is_small and len(filt) > self.max_slices:
                idx = np.unique(np.linspace(0, len(filt) - 1, self.max_slices).round().astype(int))
                stats['subsampled_dropped'] += len(filt) - len(idx)
                filt = [filt[i] for i in idx]

            kept[label_id] = [z for z, _ in filt]
            stats['final'] += len(filt)
        return kept, stats


    def process_case(self, case_dir, protected):
        t0 = time.time()
        ct_grid, ct_np, mr_np, _, qc = self.prepare_case(case_dir)
        t_reg = time.time() - t0

        mr_signal = {int(z): bool(v) for z, v in
                     enumerate(mr_np.reshape(mr_np.shape[0], -1).max(axis=1) > 0)}


        kept, kept_masks = {}, {}
        stats = defaultdict(int)
        samples_orig = 0
        for label_id, seg_path in index_label_files(case_dir, self.label_names):
            if seg_path is None:
                qc.setdefault('missing', '')
                qc['missing'] = (qc['missing'] + ' ' + self.label_names[label_id]).strip()
                continue
            seg_r = sitk.Resample(sitk.ReadImage(str(seg_path)), ct_grid, sitk.Transform(),
                                  sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            mask = sitk.GetArrayFromImage(seg_r) > 0
            areas = mask.reshape(mask.shape[0], -1).sum(axis=1)
            zs = np.flatnonzero(areas)
            samples_orig += int(zs.size)
            if zs.size == 0:
                continue
            items = {label_id: [(int(z), int(areas[z])) for z in zs]}
            organ_kept, organ_stats = self.select_slices(items, mr_signal, protected)
            for k, v in organ_stats.items():
                stats[k] += v
            if label_id in organ_kept:
                kept[label_id] = organ_kept[label_id]
                kept_masks[label_id] = {z: mask[z] for z in organ_kept[label_id]}
            del mask, seg_r

        qc.update({f'kept_{k}': v for k, v in stats.items()})
        qc['samples_orig'] = samples_orig
        qc['samples_kept'] = sum(len(v) for v in kept.values())
        qc['reg_time'] = round(t_reg, 1)
        qc['total_time'] = round(time.time() - t0, 1)

        if not self.dry_run:
            patient = case_dir.name
            needed = sorted({z for zs in kept.values() for z in zs})
            for z in needed:
                images = np.stack([ct_np[z][None, None], mr_np[z][None, None]], axis=0).astype(np.float32)
                np.savez_compressed(self.slices_dir / f'{patient}_s{z:04d}.npz', images=images)
            for label_id, zs in kept.items():
                for z in zs:
                    np.savez_compressed(
                        self.samples_dir / str(label_id) / f'{patient}_s{z:04d}.npz',
                        labels=kept_masks[label_id][z][None, None].astype(np.uint8),
                        label_id=label_id,
                        image_path=f'slices/{patient}_s{z:04d}.npz')
        return qc


    def run(self, max_cases=None):
        cases = sorted(p for p in self.raw_root.iterdir() if p.is_dir())
        if max_cases:
            cases = cases[:max_cases]

        print(f'[1/3] 统计各器官切片数（共 {len(cases)} 例）...', flush=True)
        per_organ, _ = count_organ_slices(self.raw_root, self.label_names, cases)
        protected = {l: c < self.protect_below for l, c in per_organ.items()}
        n_prot = sum(protected.values())
        print(f'      小器官（<{self.protect_below} 样本，全量保留）: {n_prot}/{len(per_organ)}')
        for l in sorted(per_organ):
            print(f'        {self.label_names[l]:<18}{per_organ[l]:>6}'
                  f'{"  [保护]" if protected[l] else ""}')
        if not self.dry_run:
            (self.out_root / 'organ_counts.json').write_text(
                json.dumps({'counts': {self.label_names[l]: c for l, c in per_organ.items()},
                            'protected': [self.label_names[l] for l, v in protected.items() if v]},
                           ensure_ascii=False, indent=2), encoding='utf-8')

        print(f'\n[2/3] 逐例配准 + 裁剪（registration={self.registration}, '
              f'max_slices={self.max_slices}, dry_run={self.dry_run}）...', flush=True)
        rows, totals = [], Counter()
        for i, case_dir in enumerate(cases, 1):
            qc = self.process_case(case_dir, protected)
            qc['patient'] = case_dir.name
            rows.append(qc)
            for k in ('samples_orig', 'samples_kept', 'sliver_dropped', 'mr_empty_dropped',
                      'mr_empty_kept', 'subsampled_dropped'):
                totals[k] += qc.get(k, qc.get(f'kept_{k}', 0))
            print(f'  [{i}/{len(cases)}] {case_dir.name}: 配准 {qc["reg_time"]:>5.1f}s  '
                  f'dice {qc["dice_before"]:.2f}->{qc["dice_final"]:.2f} ({qc["deform"]})  '
                  f'样本 {qc["samples_orig"]}->{qc["samples_kept"]}', flush=True)

        if rows:
            self.save_csv(rows, self.out_root / 'registration_qc.csv')
            self.save_csv(rows, self.out_root / 'filter_qc.csv')
        print(f'\n样本合计: {totals["samples_orig"]} -> {totals["samples_kept"]} '
              f'({totals["samples_kept"] / max(totals["samples_orig"], 1):.1%})')
        print(f'  碎屑 -{totals["sliver_dropped"]}  MR空层 -{totals["mr_empty_dropped"]} '
              f'(保留 {totals["mr_empty_kept"]})  抽样 -{totals["subsampled_dropped"]}')
        if not self.dry_run:
            print('\n[3/3] 划分 train / val / test ...', flush=True)
            self.split()

    def save_csv(self, rows, path):
        keys = sorted({k for r in rows for k in r})
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f'QC 已写入 {path}')

    def split(self):
        samples = sorted(self.samples_dir.glob('*/*.npz'))
        patients = sorted({p.stem.rsplit('_s', 1)[0] for p in samples})
        if getattr(self, 'split_file', None) is None:
            shuffled = list(patients)
            random.Random(self.seed).shuffle(shuffled)
            n_val = int(round(len(shuffled) * self.val_ratio))
            n_test = int(round(len(shuffled) * self.test_ratio))
            splits = {'val': shuffled[:n_val],
                      'test': shuffled[n_val:n_val + n_test],
                      'train': shuffled[n_val + n_test:]}
        else:
            splits = json.loads(self.split_file.read_text(encoding='utf-8'))
            if set(splits) != {'train', 'val', 'test'}:
                raise ValueError('Expected train, val and test lists.')
            assigned = [case for group in splits.values() for case in group]
            if len(assigned) != len(set(assigned)) or set(assigned) != set(patients):
                raise ValueError('Split must assign every prepared patient exactly once.')
        p2s = {p: name for name, ps in splits.items() for p in ps}

        for name in ('train', 'val', 'test'):
            (self.out_root / name).mkdir(parents=True, exist_ok=True)
            for old in (self.out_root / name).glob('*.npz'):
                old.unlink()

        counter, per_organ = Counter(), defaultdict(Counter)
        for sample in samples:
            label_id = int(sample.parent.name)
            split = p2s[sample.stem.rsplit('_s', 1)[0]]
            dst = self.out_root / split / f'l{label_id:02d}_{sample.name}'
            if dst.exists():
                dst.unlink()
            try:
                os.link(sample, dst)
            except OSError:
                shutil.copy2(sample, dst)
            counter[split] += 1
            per_organ[label_id][split] += 1

        (self.out_root / 'split_patients.json').write_text(
            json.dumps(splits, ensure_ascii=False, indent=2), encoding='utf-8')
        with open(self.out_root / 'split_patients.txt', 'w', encoding='utf-8') as f:
            for name in ('train', 'val', 'test'):
                f.write(f'[{name}] {len(splits[name])} 个病人\n  ' + ' '.join(splits[name]) + '\n')
            f.write('\n每个器官的样本数（train / val / test）:\n')
            for label_id in sorted(per_organ):
                c = per_organ[label_id]
                f.write(f'  {self.label_names[label_id]:<18} {c["train"]} / {c["val"]} / {c["test"]}\n')

        print('病人数:', {k: len(splits[k]) for k in ('train', 'val', 'test')})
        print('样本数:', dict(counter))


def main():
    p = argparse.ArgumentParser(description='HaN-Seg 逐层样本生成 v2：智能裁剪 + 可选非刚体配准')
    p.add_argument('--raw', type=Path, default=Path(r'G:\datesets\HaNSeg\raw\HaN-Seg\set_1'))
    p.add_argument('--out', type=Path, default=Path(r'G:\datesets\HaNSeg\SI_v2'))
    p.add_argument('--yaml', type=Path, default=Path(__file__).resolve().parents[1] / 'config' / 'nanseg.yaml')
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--registration', choices=['rigid', 'deformable'], default='rigid')
    p.add_argument('--max-slices', type=int, default=20, )
    p.add_argument('--protect-below', type=int, default=400)
    p.add_argument('--min-rel-area', type=float, default=0.02)
    p.add_argument('--min-area-px', type=int, default=4)
    p.add_argument('--keep-mr-empty', action='store_true')
    p.add_argument('--max-deform-mm', type=float, default=25.0)
    p.add_argument('--min-gain', type=float, default=0.0)
    p.add_argument('--val-ratio', type=float, default=0.15)
    p.add_argument('--test-ratio', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cases', type=int, default=None)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--split-file', type=Path)
    args = p.parse_args()

    for path in (args.raw, args.out):
        if not str(path).isascii():
            raise SystemExit(f'路径含非 ASCII 字符，SimpleITK 无法读取: {path}')
    if not args.raw.is_dir():
        raise SystemExit(f'找不到原始数据目录: {args.raw}')

    writer = HaNSegWriterV2(
        raw_root=args.raw, out_root=args.out, label_names=read_labels(args.yaml),
        size=args.size, registration=args.registration, max_slices=args.max_slices,
        protect_below=args.protect_below, min_rel_area=args.min_rel_area,
        min_area_px=args.min_area_px, drop_mr_empty=not args.keep_mr_empty,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed,
        max_deform_mm=args.max_deform_mm, min_gain=args.min_gain, dry_run=args.dry_run)
    writer.split_file = args.split_file
    writer.run(max_cases=args.cases)


if __name__ == '__main__':
    main()
