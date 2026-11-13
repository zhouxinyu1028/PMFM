import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml


def read_labels(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['labels']


def otsu_threshold(image, bins=256):

    values = image[np.isfinite(image)]
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return low
    hist, edges = np.histogram(values, bins=bins, range=(low, high))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = weight_bg[-1] - weight_bg
    cum = np.cumsum(hist * centers)
    mean_bg = cum / np.maximum(weight_bg, 1e-9)
    mean_fg = (cum[-1] - cum) / np.maximum(weight_fg, 1e-9)
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    return float(centers[int(np.argmax(between))])


def body_dice(ct_array, mr_array):

    covered = mr_array.reshape(mr_array.shape[0], -1).max(axis=1) > 0
    if not covered.any():
        return 0.0
    ct_array, mr_array = ct_array[covered], mr_array[covered]
    ct_body = ct_array > -300
    mr_body = mr_array > otsu_threshold(mr_array)
    if not ct_body.any() or not mr_body.any():
        return 0.0
    return float(2.0 * np.logical_and(ct_body, mr_body).sum() / (ct_body.sum() + mr_body.sum()))


def shrink_factor(spacing, target_mm):
    return max(1, int(round(target_mm / float(spacing))))


def decode_mr(mr):

    if int(sitk.GetArrayViewFromImage(mr).min()) >= 32768:
        return sitk.Clamp(sitk.Cast(mr, sitk.sitkFloat32) - 32768.0, lowerBound=0.0)
    return sitk.Cast(mr, sitk.sitkFloat32)


def register_mr_to_ct(ct, mr, init_filter=sitk.CenteredTransformInitializerFilter.GEOMETRY):

    ct_coarse = sitk.Cast(sitk.Shrink(ct, [shrink_factor(ct.GetSpacing()[0], 2.0),
                                           shrink_factor(ct.GetSpacing()[1], 2.0),
                                           shrink_factor(ct.GetSpacing()[2], 4.0)]), sitk.sitkFloat32)
    mr_coarse = sitk.Cast(sitk.Shrink(mr, [shrink_factor(mr.GetSpacing()[0], 2.0),
                                           shrink_factor(mr.GetSpacing()[1], 2.0),
                                           shrink_factor(mr.GetSpacing()[2], 4.0)]), sitk.sitkFloat32)

    initial = sitk.CenteredTransformInitializer(ct_coarse, mr_coarse, sitk.Euler3DTransform(), init_filter)

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.2, seed=42)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-5,
                                                          numberOfIterations=300, relaxationFactor=0.5)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([2, 1])
    registration.SetSmoothingSigmasPerLevel([1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial, inPlace=False)

    transform = registration.Execute(ct_coarse, mr_coarse)
    return transform, float(registration.GetMetricValue()), registration.GetOptimizerStopConditionDescription()


def index_label_files(case_dir, label_names):

    files = {}
    for path in case_dir.glob('*_OAR_*.seg.nrrd'):
        name = path.name.split('_OAR_', 1)[1].replace('.seg.nrrd', '')
        files[name] = path
    return [(label_id, files.get(name)) for label_id, name in enumerate(label_names)]


class HaNSegSliceWriter:
    def __init__(self, raw_root, out_root, label_names, size=512):
        self.raw_root = Path(raw_root)
        self.out_root = Path(out_root)
        self.label_names = label_names
        self.size = size
        self.slices_dir = self.out_root / 'slices'
        self.samples_dir = self.out_root / 'samples'
        self.slices_dir.mkdir(parents=True, exist_ok=True)
        for label_id in range(len(label_names)):
            (self.samples_dir / str(label_id)).mkdir(parents=True, exist_ok=True)

    def run(self, max_cases=None):
        cases = sorted(p for p in self.raw_root.iterdir() if p.is_dir())
        if max_cases:
            cases = cases[:max_cases]
        total = 0
        qc_rows = []
        for i, case_dir in enumerate(cases, 1):
            count, qc = self.process_case(case_dir)
            qc_rows.append(qc)
            total += count
            print(f'[{i}/{len(cases)}] {case_dir.name}: {count} 个器官样本，'
                  f'体部重合度 {qc["dice_before"]:.2f} -> {qc["dice_after"]:.2f}，'
                  f'MR 覆盖 {qc["mr_cover"]:.2f}（累计 {total}）', flush=True)
        self.save_qc(qc_rows)

    def save_qc(self, rows):
        if not rows:
            return
        path = self.out_root / 'registration_qc.csv'
        with open(path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f'配准/覆盖情况已写入 {path}')

    def process_case(self, case_dir):
        ct = sitk.ReadImage(str(next(case_dir.glob('*_IMG_CT.nrrd'))))
        mr = decode_mr(sitk.ReadImage(str(next(case_dir.glob('*_IMG_MR*.nrrd')))))

        target_size = (self.size, self.size, ct.GetSize()[2])
        spacing = np.asarray(ct.GetSpacing(), dtype=np.float64)
        target_spacing = np.array([spacing[0] * ct.GetSize()[0] / self.size,
                                   spacing[1] * ct.GetSize()[1] / self.size,
                                   spacing[2]])

        direction = np.asarray(ct.GetDirection(), dtype=np.float64).reshape(3, 3)
        target_origin = np.asarray(ct.GetOrigin(), dtype=np.float64) + direction @ (0.5 * (target_spacing - spacing))

        ct_resampled = sitk.Resample(ct, target_size, sitk.Transform(), sitk.sitkLinear,
                                     tuple(target_origin), tuple(target_spacing), ct.GetDirection(),
                                     0.0, sitk.sitkFloat32)
        ct_np = sitk.GetArrayFromImage(ct_resampled)  # (s, 512, 512)

        mr_raw = sitk.GetArrayFromImage(
            sitk.Resample(mr, ct_resampled, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32))
        dice_before = body_dice(ct_np, mr_raw)


        best = None
        for init_name, init_filter in (('GEOMETRY', sitk.CenteredTransformInitializerFilter.GEOMETRY),
                                       ('MOMENTS', sitk.CenteredTransformInitializerFilter.MOMENTS)):
            try:
                transform, metric, stop = register_mr_to_ct(ct, mr, init_filter)
            except RuntimeError as exc:
                print(f'  配准({init_name})失败: {exc}', flush=True)
                continue
            candidate = sitk.GetArrayFromImage(
                sitk.Resample(mr, ct_resampled, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32))
            dice = body_dice(ct_np, candidate)
            if best is None or dice > best[0]:
                best = (dice, transform, metric, stop, candidate, init_name)

        if best is None:  
            dice_after, transform, metric, stop, mr_np, init_name = dice_before, sitk.Transform(3, sitk.sitkIdentity), float('nan'), 'failed', mr_raw, 'none'
        else:
            dice_after, transform, metric, stop, mr_np, init_name = best

        qc = {
            'patient': case_dir.name,
            'dice_before': round(dice_before, 4),
            'dice_after': round(dice_after, 4),
            'mr_cover': round(float((mr_np.reshape(mr_np.shape[0], -1).max(axis=1) > 0).mean()), 4),
            'init': init_name,
            'metric': metric,
            'stop': stop,
            'params': ' '.join(f'{p:.4f}' for p in np.ravel(transform.GetParameters())) if hasattr(transform, 'GetParameters') else '',
            'missing': '',
        }

        patient = case_dir.name
        needed_slices = np.zeros(ct_np.shape[0], dtype=bool)
        count = 0

        for label_id, seg_path in index_label_files(case_dir, self.label_names):
            if seg_path is None:  # 个别病例没有标注该器官（如 case_19 无 OpticChiasm）
                qc['missing'] = (qc['missing'] + ' ' + self.label_names[label_id]).strip()
                continue
            seg = sitk.ReadImage(str(seg_path))
            seg_resampled = sitk.Resample(seg, ct_resampled, sitk.Transform(), sitk.sitkNearestNeighbor,
                                          0, sitk.sitkUInt8)
            mask = sitk.GetArrayFromImage(seg_resampled) > 0  # (s, 512, 512)
            slices = np.flatnonzero(mask.reshape(mask.shape[0], -1).any(axis=1))
            for z in slices:
                np.savez_compressed(self.samples_dir / str(label_id) / f'{patient}_s{z:04d}.npz',
                                    labels=mask[z][None, None].astype(np.uint8),
                                    label_id=label_id,
                                    image_path=f'slices/{patient}_s{z:04d}.npz')
            needed_slices[slices] = True
            count += slices.size

        if qc['missing']:
            print(f'  {patient} 缺少器官标注: {qc["missing"]}（该器官本病例无样本）', flush=True)

        for z in np.flatnonzero(needed_slices):
            images = np.stack([ct_np[z][None, None], mr_np[z][None, None]], axis=0)
            np.savez_compressed(self.slices_dir / f'{patient}_s{z:04d}.npz', images=images)
        return count, qc


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    label_names = read_labels(project_root / 'config' / 'nanseg.yaml')


    raw_root = Path(r'G:\datesets\HaNSeg\raw\HaN-Seg\set_1')   # 原始 nrrd 目录（必须是纯英文路径）
    out_root = Path(r'G:\datesets\HaNSeg\SI')                  # 预处理结果根目录

    for path in (raw_root, out_root):
        if not str(path).isascii():
            raise SystemExit(f'路径含非 ASCII 字符，SimpleITK 无法读取: {path}\n'
                             f'请把原始数据放到纯英文路径下，例如 G:\\datesets\\HaNSeg\\raw\\HaN-Seg')
    if not raw_root.is_dir():
        raise SystemExit(f'找不到原始数据目录: {raw_root}')

    HaNSegSliceWriter(raw_root, out_root, label_names, size=512).run()
