from __future__ import annotations

import argparse
import sys
import csv
import re
import shutil
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


import cv2
import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

from evaluation.hanseg_metrics import (BILATERAL_GROUPS, SUMMARY_FIELDS, binary_metrics,
                                       format_mean_std, merge_bilateral, summarize)
from model.builder_model import build_model
from utils.load_yaml_file import read_yaml
from utils.transforms2D import Crop

ROOT = Path(__file__).resolve().parent
YAML = read_yaml(ROOT / 'config' / 'nanseg.yaml')
LABELS = YAML['labels']

SLICES_DIR = Path(YAML['data']['val']['datas_dir']).parent / 'slices'
TEST_DIR = Path(YAML['data']['val']['datas_dir']).parent / 'test'
RAW_ROOT = ROOT / 'data' / 'raw' / 'set_1'
SIZE = 512
CASE_RE = re.compile(r'(case_\d+)')
Z_RE = re.compile(r'_s(\d+)\.npz$')


@contextmanager
def sitk_path(path):
    if str(path).isascii():
        yield str(path)
        return
    with TemporaryDirectory(dir=ROOT if str(ROOT).isascii() else None) as directory:
        staged = Path(directory) / 'image.nrrd'
        shutil.copyfile(path, staged)
        yield str(staged)


def discover_cases(sample_dir: Path) -> list[str]:
    """从测试样本名（l00_case_05_s0033.npz）里解析病例号。"""
    cases = {m.group(1) for p in sample_dir.glob('*.npz') if (m := CASE_RE.search(p.name))}
    return sorted(cases)


def build_models(ckpt_path: Path, device: torch.device):
    unet = build_model(YAML['model']['backbone']).to(device)
    pff = build_model(YAML['model']['fusion']).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    unet.load_state_dict(state['unet'])
    pff.load_state_dict(state['pff'])
    unet.eval()
    pff.eval()
    return unet, pff, state.get('epoch')


class CaseGrid:
    """预处理后的 512 网格：spacing=(x_mm, y_mm, z_mm)，数组轴序为 (z, y, x)。"""

    def __init__(self, raw_root: Path, case: str, size: int = SIZE):
        self.case = case
        self.raw_root = raw_root
        self.size = size
        ct_files = sorted((raw_root / case).glob('*_IMG_CT.nrrd'))
        if not ct_files:
            raise FileNotFoundError(f'找不到 {case} 的 CT NRRD（{raw_root / case}）')
        reader = sitk.ImageFileReader()
        with sitk_path(ct_files[0]) as filename:
            reader.SetFileName(filename)
            reader.ReadImageInformation()
        self.origin = np.asarray(reader.GetOrigin(), dtype=np.float64)
        self.direction_tuple = reader.GetDirection()
        self.direction = np.asarray(self.direction_tuple).reshape(3, 3)
        native = np.asarray(reader.GetSpacing(), dtype=np.float64)
        native_size = np.asarray(reader.GetSize(), dtype=np.float64)
        self.spacing_xyz = np.array([native[0] * native_size[0] / size,
                                     native[1] * native_size[1] / size,
                                     native[2]])
        self.n_slices = int(native_size[2])
        self.target_origin = self.origin + self.direction @ (0.5 * (self.spacing_xyz - native))

    def gt_volume(self, organ: str):
        """返回 (bool 数组 (z, y, x), spacing (z, y, x) mm)；器官不存在时返回 (None, spacing)。"""
        spacing_zyx = (float(self.spacing_xyz[2]), float(self.spacing_xyz[1]),
                       float(self.spacing_xyz[0]))
        seg_path = self.raw_root / self.case / f'{self.case}_OAR_{organ}.seg.nrrd'
        if not seg_path.exists():
            return None, spacing_zyx
        reference = sitk.Image((self.size, self.size, self.n_slices), sitk.sitkFloat32)
        reference.SetSpacing(tuple(self.spacing_xyz))
        reference.SetOrigin(tuple(self.target_origin))
        reference.SetDirection(self.direction_tuple)
        with sitk_path(seg_path) as filename:
            segmentation = sitk.ReadImage(filename)
        resampled = sitk.Resample(segmentation, reference, sitk.Transform(),
                                  sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        return sitk.GetArrayFromImage(resampled) > 0, spacing_zyx


def normalize(array: np.ndarray) -> np.ndarray:
    array = array - array.min()
    peak = array.max()
    return array / peak if peak > 0 else array * 0


@torch.no_grad()
def predict_case_slices(unet, pff, case: str, zs: list[int], gt_volumes: dict,
                        device, slices_dir: Path = SLICES_DIR):
    """只对器官 GT 为正的切片推理，返回 {organ_id: (len(zs), 512, 512) uint8}。"""
    boxes = {oid: np.zeros((len(zs), SIZE, SIZE), dtype=np.uint8) for oid in gt_volumes}
    crop = Crop(int(YAML['data']['transforms']['Crop_w']), int(YAML['data']['transforms']['Crop_h']),
                int(YAML['data']['transforms']['step']), int(YAML['data']['transforms']['eq_cnt']))
    cached_z, cached_images = None, None
    tasks = [(i, z, oid) for i, z in enumerate(zs) for oid, vol in gt_volumes.items() if vol[i].any()]
    for i, z, organ_id in tqdm(tasks, desc=f'infer {case}', unit='slice'):
        if z != cached_z:
            cached_z, cached_images = z, np.load(slices_dir / f'{case}_s{z:04d}.npz')['images']
        images = cached_images
        ct_full = images[0, 0, 0].astype(np.float32)
        mr_full = images[1, 0, 0].astype(np.float32)
        gt_slice = gt_volumes[organ_id][i].astype(np.float32)
        ct_crop, mr_crop, _ = crop(torch.from_numpy(ct_full)[None, None],
                                   torch.from_numpy(mr_full)[None, None],
                                   torch.from_numpy(gt_slice)[None, None])
        ct_raw = ct_crop[0, 0].numpy().astype(np.float32)
        ct_in = normalize(ct_crop[0])[None].float().to(device)
        mr_in = normalize(mr_crop[0])[None].float().to(device)
        ct_pred, point_ct, ct_last, _ = unet(ct_in)
        mr_pred, point_mr, mr_last, _ = unet(mr_in)
        fusion = pff(point_ct, point_mr, ct_last, mr_last, ct_pred, mr_pred)
        pred = (torch.sigmoid(fusion[:, organ_id:organ_id + 1]) > 0.5)[0, 0].cpu().numpy().astype(np.uint8)
        _, _, loc, _ = cv2.minMaxLoc(cv2.matchTemplate(ct_full, ct_raw, cv2.TM_SQDIFF))
        x0, y0 = loc
        boxes[organ_id][i, y0:y0 + pred.shape[0], x0:x0 + pred.shape[1]] = pred
    return boxes


def summarize_organs(per_organ: OrderedDict) -> list[OrderedDict]:
    """逐器官 mean ± std；合并行（左右成对器官）按 Table 2 口径一并给出。"""
    rows = []
    for name, cases in per_organ.items():
        row = OrderedDict(organ=name)
        row.update(summarize(cases))
        row['empty_prediction'] = sum(1 for c in cases
                                      if c.status == 'empty_prediction' and c.evaluated)
        row['gt_empty'] = sum(1 for c in cases if c.status in ('gt_empty', 'both_empty'))
        row['bilateral'] = name in BILATERAL_GROUPS
        rows.append(row)
    return rows


def print_table(rows, ascii_symbols=False):
    header = (f"{'organ':<20}{'N':>4}{'DC (%)':>20}{'HD95 (mm)':>18}"
              f"{'Jaccard':>18}{'ASSD (mm)':>18}{'miss':>6}{'noGT':>6}")
    print('\n' + header)
    print('-' * len(header))
    for r in rows:
        label = r['organ'] + (' (L+R)' if r['bilateral'] else '')
        line = (f"{label:<20}{r['n']:>4}"
              f"{format_mean_std(r['dice_mean'] * 100, r['dice_std'] * 100, 2):>20}"
              f"{format_mean_std(r['hd95_mean'], r['hd95_std'], 2):>18}"
              f"{format_mean_std(r['jaccard_mean'], r['jaccard_std'], 4):>18}"
              f"{format_mean_std(r['assd_mean'], r['assd_std'], 2):>18}"
              f"{r['empty_prediction']:>6}{r['gt_empty']:>6}")
        print(line.replace('\u00b1', '+/-') if ascii_symbols else line)


def main():
    parser = argparse.ArgumentParser(description='HaN-Seg 3D 逐器官评估（mean ± std）')
    parser.add_argument('--ckpt', type=Path,
                        default=ROOT / 'weights' / 'latest_model.pth')
    parser.add_argument('--cases', default='', help='逗号分隔的病例号，默认取 test 里的全部病例')
    parser.add_argument('--organs', default='', help='逗号分隔的器官 id，默认全部')
    parser.add_argument('--out', type=Path, default=None, help='逐器官汇总 CSV')
    parser.add_argument('--show-saved', action='store_true', help='读取 --out 指定的已有 CSV，按原始表格格式显示')
    parser.add_argument('--ascii', action='store_true', help='用 +/- 代替 ±，兼容旧终端编码')
    parser.add_argument('--per-case-out', type=Path, default=None, help='逐病例明细 CSV')
    parser.add_argument('--raw-root', type=Path, default=RAW_ROOT)
    parser.add_argument('--slices-dir', type=Path, default=SLICES_DIR)
    parser.add_argument('--test-dir', '--data-dir', dest='test_dir', type=Path, default=TEST_DIR)
    parser.add_argument('--hd95-mode', choices=['max', 'pooled'], default='max',
                        help="HD95 定义：'max' 两方向 95 分位取最大（标准），'pooled' 为 MedPy 定义")
    parser.add_argument('--empty-policy', choices=['penalty', 'nan', 'skip'], default='penalty',
                        help='GT 非空而预测为空时：penalty 记体积对角线，nan 记 nan，skip 剔除该病例')
    args = parser.parse_args()

    cases = [c.strip() for c in args.cases.split(',') if c.strip()] or discover_cases(args.test_dir)
    organ_ids = ([int(x) for x in args.organs.split(',') if x.strip()]
                 if args.organs else list(range(len(LABELS))))
    out_path = args.out or ROOT / 'weights' / f'metrics_3d_{args.ckpt.stem}.csv'
    per_case_path = args.per_case_out or out_path.with_name(out_path.stem + '_per_case.csv')

    if args.show_saved:
        with out_path.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            for key in SUMMARY_FIELDS:
                row[key] = int(row[key]) if key == 'n' else float(row[key])
            for key in ('empty_prediction', 'gt_empty'):
                row[key] = int(row[key])
            row['bilateral'] = str(row['bilateral']).strip().lower() in ('true', '1')
        print(f'Saved evaluation results (no new inference): {out_path}')
        print_table(rows, ascii_symbols=args.ascii)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet, pff, epoch = build_models(args.ckpt, device)
    print(f'checkpoint {args.ckpt.name}, epoch={epoch}, device={device}')
    print(f'cases={cases} hd95_mode={args.hd95_mode} empty_policy={args.empty_policy}')

    per_organ: OrderedDict = OrderedDict((LABELS[o], []) for o in organ_ids)
    records = []
    for case in cases:
        grid = CaseGrid(args.raw_root, case)
        zs = sorted(int(Z_RE.search(p.name).group(1)) for p in args.slices_dir.glob(f'{case}_s*.npz'))
        if not zs or len(set(zs)) != len(zs) or zs[0] < 0 or zs[-1] >= grid.n_slices:
            raise ValueError(f'{case}: invalid retained slice indices')
        missing = np.ones(grid.n_slices, dtype=bool)
        missing[zs] = False
        print(f'{case}: {len(zs)} slices available, z={zs[0]}..{zs[-1]}, '
              f'spacing={grid.spacing_xyz[0]:.3f}x{grid.spacing_xyz[1]:.3f}x{grid.spacing_xyz[2]:.2f} mm')
        gt_volumes, spacings = {}, {}
        for organ_id in organ_ids:
            gt, spacing = grid.gt_volume(LABELS[organ_id])
            if gt is None:
                gt = np.zeros((grid.n_slices, SIZE, SIZE), dtype=bool)
            gt_volumes[organ_id] = gt[zs]
            spacings[organ_id] = spacing
        pred_volumes = predict_case_slices(unet, pff, case, zs, gt_volumes, device, args.slices_dir)
        for organ_id in organ_ids:
            gt_full, _ = grid.gt_volume(LABELS[organ_id])
            if gt_full is None:
                gt_full = np.zeros((grid.n_slices, SIZE, SIZE), dtype=bool)
            pred_full = np.zeros_like(gt_full)
            pred_full[zs] = pred_volumes[organ_id]
            omitted_gt = int(np.count_nonzero(gt_full[missing]))
            metrics = binary_metrics(pred_full, gt_full,
                                     spacings[organ_id], hd95_mode=args.hd95_mode,
                                     empty_policy=args.empty_policy)
            per_organ[LABELS[organ_id]].append(metrics)
            records.append(OrderedDict(case=case, organ=LABELS[organ_id],
                                       retained_slices=len(zs), full_slices=grid.n_slices,
                                       omitted_gt_voxels=omitted_gt,
                                       spacing_z=spacings[organ_id][0],
                                       spacing_y=spacings[organ_id][1],
                                       spacing_x=spacings[organ_id][2],
                                       status=metrics.status, evaluated=metrics.evaluated,
                                       dice=metrics.dice, jaccard=metrics.jaccard,
                                       hd95=metrics.hd95, hd95_pooled=metrics.hd95_pooled,
                                       assd=metrics.assd))
        del gt_volumes, pred_volumes

    grouped = OrderedDict(per_organ)
    grouped.update(merge_bilateral(per_organ))
    rows = summarize_organs(grouped)

    macro_pool = [c for cases_ in per_organ.values() for c in cases_]
    macro = OrderedDict(organ='MACRO(over organs)')
    macro.update(summarize(macro_pool))
    macro['empty_prediction'] = sum(1 for c in macro_pool if c.status == 'empty_prediction')
    macro['gt_empty'] = sum(1 for c in macro_pool if c.status in ('gt_empty', 'both_empty'))
    macro['bilateral'] = False
    rows.append(macro)
    print_table(rows, ascii_symbols=args.ascii)

    for path, fields, data in (
            (out_path, ['organ'] + SUMMARY_FIELDS + ['empty_prediction', 'gt_empty', 'bilateral'], rows),
            (per_case_path, list(records[0].keys()) if records else ['case'], records)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
        print('CSV saved to:', path)


if __name__ == '__main__':
    main()
