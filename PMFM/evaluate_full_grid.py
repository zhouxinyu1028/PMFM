"""HaN-Seg 完整体积 3D 逐器官评估代码示例。



用法
----
  python evaluate_full_grid.py --ckpt weight/nanseg_v2_t30/latest_model.pth
  python evaluate_full_grid.py --ckpt ... --tile 128 --stride 96 --batch 8
  python evaluate_full_grid.py --ckpt ... --cases case_05,case_12 --organs 5,10
  python evaluate_full_grid.py --show-saved --out metrics_3d.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluation.hanseg_metrics import (BILATERAL_GROUPS, SUMMARY_FIELDS, binary_metrics,  
                                       format_mean_std, merge_bilateral, summarize)
from model.builder_model import build_model                                              
from utils.load_yaml_file import read_yaml                                               


def _first_existing(*candidates: Path) -> Path:
    """在多个候选根目录里取第一个存在的（兼容 <root>/data/raw/set_1 与 <root>/HaN-Seg.zip/...）。"""
    for cand in candidates:
        if Path(cand).is_dir():
            return Path(cand)
    return Path(candidates[0])


YAML = read_yaml(ROOT / 'config' / 'nanseg.yaml')
LABELS = YAML['labels']
N_ORGANS = len(LABELS)


def load_config(path: Path) -> None:
    
    global YAML, LABELS, N_ORGANS
    YAML = read_yaml(Path(path))
    LABELS = YAML['labels']
    N_ORGANS = len(LABELS)

_DATASET_ROOT = Path(YAML['data']['val']['datas_dir']).parent
SLICES_DIR = _DATASET_ROOT / 'slices'
TEST_DIR = _DATASET_ROOT / 'test'
RAW_ROOT = _first_existing(ROOT / 'data' / 'raw' / 'set_1',
                           ROOT / 'HaN-Seg.zip' / 'HaN-Seg' / 'set_1')
CHECKPOINT_DIR = _first_existing(ROOT / 'weight', ROOT / 'weights')
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
    """从样本名（l00_case_05_s0033.npz / case_05_s0033.npz）里解析病例号。"""
    cases = {m.group(1) for p in Path(sample_dir).glob('*.npz') if (m := CASE_RE.search(p.name))}
    return sorted(cases)


def build_models(ckpt_path: Path, device: torch.device):
    """按**权重本身**的通道数装配模型，不依赖 YAML 里的 in_channels / num_classes。
    
    """
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    unet_sd = state['unet']
    in_ch = int(unet_sd['in_conv.double_conv.0.weight'].shape[1])
    out_ch = None
    for key, val in unet_sd.items():
        if key.startswith('out_conv') and val.dim() == 1:
            out_ch = int(val.shape[0])
            break
    out_ch = out_ch or len(LABELS)

    backbone = dict(YAML['model']['backbone'])
    backbone['in_channels'] = in_ch
    backbone['num_classes'] = out_ch
    fusion = dict(YAML['model']['fusion'])
    fusion['num_classes'] = out_ch
    unet = build_model(backbone).to(device)
    pff = build_model(fusion).to(device)
    unet.load_state_dict(unet_sd)
    pff.load_state_dict(state['pff'])
    unet.eval()
    pff.eval()
    return unet, pff, state.get('epoch'), in_ch, out_ch


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


def window_starts(length: int, tile: int, stride: int) -> list[int]:
    """铺满 [0, length) 的窗口起点；最后一个窗口贴边，保证整层被覆盖。"""
    if tile >= length:
        return [0]
    stride = max(int(stride), 1)
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def hann_weight(tile: int) -> np.ndarray:
    """2D Hann 融合权重。下限 1e-2：Hann 在四角为 0，直接除会让边缘像素被噪声放大。"""
    w = np.hanning(tile).astype(np.float32)
    return np.maximum(np.outer(w, w), 1e-2)


@torch.no_grad()
def predict_case_sliding(unet, pff, case: str, zs: list[int], device, get_slice,
                         n_organs: int = N_ORGANS,
                         tile: int = 128, stride: int = 96, thr: float = 0.5,
                         batch: int = 8, skip_air: bool = True, air_ratio: float = 0.02):
    """整层滑窗推理**。

    对 zs 里每一层：在 512² 上按 tile/stride 铺满窗口 → 每个窗口各自 min-max 归一化
    （与训练一致，训练窗口就是 tile×tile 的裁片）→ 一次前向出全部 n_organs 通道 →
    Hann 加权累加 → 除权重 → 阈值化。

    """
    size = SIZE
    step_ys = window_starts(size, tile, stride)
    step_xs = window_starts(size, tile, stride)
    weight = hann_weight(tile)
    packed_len = (size * size) // 8
    boxes = OrderedDict((oid, np.zeros((len(zs), packed_len), dtype=np.uint8))
                        for oid in range(n_organs))
    n_skipped = 0
    n_windows = 0

    for i, z in enumerate(tqdm(zs, desc=f'infer {case}', unit='slice', leave=False)):
        ct_full, mr_full = get_slice(z)
        ct_full = np.asarray(ct_full, dtype=np.float32)
        mr_full = np.asarray(mr_full, dtype=np.float32)
        ct_peak = float(ct_full.max())

        prob_acc = np.zeros((n_organs, size, size), dtype=np.float32)
        weight_acc = np.zeros((size, size), dtype=np.float32)
        pending: list[tuple[int, int]] = []

        def flush():
            if not pending:
                return
            ct_b = np.stack([normalize(ct_full[y:y + tile, x:x + tile]) for y, x in pending])
            mr_b = np.stack([normalize(mr_full[y:y + tile, x:x + tile]) for y, x in pending])
            ct_t = torch.from_numpy(ct_b)[:, None].float().to(device)
            mr_t = torch.from_numpy(mr_b)[:, None].float().to(device)
            ct_pred, point_ct, ct_last, _ = unet(ct_t)
            mr_pred, point_mr, mr_last, _ = unet(mr_t)
            fusion = pff(point_ct, point_mr, ct_last, mr_last, ct_pred, mr_pred)
            probs = torch.sigmoid(fusion).cpu().numpy()          # (B, n_organs, tile, tile)
            for k, (y0, x0) in enumerate(pending):
                prob_acc[:, y0:y0 + tile, x0:x0 + tile] += probs[k] * weight
                weight_acc[y0:y0 + tile, x0:x0 + tile] += weight
            pending.clear()

        for y0 in step_ys:
            for x0 in step_xs:
                if skip_air and ct_peak > 0:
                    win_peak = float(ct_full[y0:y0 + tile, x0:x0 + tile].max())
                    if win_peak < air_ratio * ct_peak:
                        n_skipped += 1
                        continue
                pending.append((y0, x0))
                n_windows += 1
                if len(pending) >= max(int(batch), 1):
                    flush()
        flush()

        prob = prob_acc / np.maximum(weight_acc, 1e-6)
        pred = prob > thr                                       # (n_organs, 512, 512)
        for oid in range(n_organs):
            boxes[oid][i] = np.packbits(pred[oid].reshape(-1))

    if n_skipped:
        print(f'  {case}: 跳过纯空气窗口 {n_skipped} 个 / 共 {n_windows + n_skipped} 个'
              f'（{n_skipped / max(n_windows + n_skipped, 1) * 100:.1f}%，仅依据 CT 强度）')
    return boxes


def make_slice_getter(case: str, source: str = 'dir', slices_dir: Path = SLICES_DIR,
                      raw_root: Path = RAW_ROOT, cache_dir: Path | None = None):
    """返回 (get_slice, zs)：get_slice(z) -> (ct 512², mr 512²) float32。
    source='dir'    : 读预处理切片目录。
    source='volume' : 从原始 NRRD 重建完整体积。
    """
    if source == 'volume':
        from evaluate_hanseg_3d_volume import prepare_volume
        ct, mr, geom = prepare_volume(raw_root, case, cache_dir)

        def get(zz: int):
            return ct[zz].astype(np.float32), mr[zz].astype(np.float32)
        return get, list(range(int(ct.shape[0])))

    def get(zz: int):
        images = np.load(Path(slices_dir) / f'{case}_s{zz:04d}.npz')['images']
        ct = np.asarray(images[0]).reshape(-1, SIZE, SIZE)[0].astype(np.float32)
        mr = np.asarray(images[1]).reshape(-1, SIZE, SIZE)[0].astype(np.float32)
        return ct, mr

    zs = sorted(int(Z_RE.search(p.name).group(1))
                for p in Path(slices_dir).glob(f'{case}_s*.npz'))
    return get, zs


@torch.no_grad()
def predict_case_whole(unet, pff, case: str, zs: list[int], device, get_slice,
                       n_organs: int = N_ORGANS,
                       zdepth: int = 3, factor: int = 2, thr: float = 0.5, batch: int = 8):
    """整层直接推理，非滑窗，根据训练的模型而定
    """
    size = SIZE
    half = max(int(zdepth) // 2, 0)
    lo, hi = int(factor), int(factor)
    packed_len = (size * size) // 8
    boxes = OrderedDict((oid, np.zeros((len(zs), packed_len), dtype=np.uint8))
                        for oid in range(n_organs))
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def load_slice(zz: int):
        if zz not in cache:
            ct, mr = get_slice(zz)
            ct = np.asarray(ct, dtype=np.float32)
            mr = np.asarray(mr, dtype=np.float32)
            if lo > 1:
                ct = torch.nn.functional.avg_pool2d(
                    torch.from_numpy(ct)[None, None], lo)[0, 0].numpy()
                mr = torch.nn.functional.avg_pool2d(
                    torch.from_numpy(mr)[None, None], lo)[0, 0].numpy()
            cache[zz] = (ct, mr)
        return cache[zz]

    def stack(center: int):
        cts, mrs = [], []
        for dz in range(-half, half + 1):
            zz = center + dz
            if zz not in zs:                 # 缺层复制中心层（与训练一致）
                zz = center
            c, m = load_slice(zz)
            cts.append(c)
            mrs.append(m)
        return np.stack(cts), np.stack(mrs)  # (k, h, w)

    def joint(arrays: list[np.ndarray]):
        """整叠联合 min-max，与 TrainDataset.normalize_minmax 一致。"""
        t = np.stack(arrays).astype(np.float32)
        t = t - t.min()
        peak = t.max()
        return t / peak if peak > 0 else t * 0

    pending: list[int] = []
    positions: list[int] = []

    def flush():
        if not pending:
            return
        ct_b = torch.from_numpy(np.stack([joint(s[0]) for s in pending])).float().to(device)
        mr_b = torch.from_numpy(np.stack([joint(s[1]) for s in pending])).float().to(device)
        ct_pred, point_ct, ct_last, _ = unet(ct_b)
        mr_pred, point_mr, mr_last, _ = unet(mr_b)
        fusion = pff(point_ct, point_mr, ct_last, mr_last, ct_pred, mr_pred)
        probs = torch.sigmoid(fusion)                         # (B, n_organs, h, w)
        probs = torch.nn.functional.interpolate(probs, size=(size, size),
                                                mode='bilinear', align_corners=False)
        probs = probs.cpu().numpy()
        for k, i in enumerate(positions):
            pred = probs[k] > thr                            # (n_organs, 512, 512)
            for oid in range(n_organs):
                boxes[oid][i] = np.packbits(pred[oid].reshape(-1))
        pending.clear()
        positions.clear()

    for i, z in enumerate(tqdm(zs, desc=f'infer {case}', unit='slice', leave=False)):
        pending.append(stack(z))
        positions.append(i)
        if len(pending) >= max(int(batch), 1):
            flush()
    flush()
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
    parser = argparse.ArgumentParser(
        description='HaN-Seg 完整体积 3D 逐器官评估')
    parser.add_argument('--ckpt', type=Path, default=CHECKPOINT_DIR / 'latest_model.pth')
    parser.add_argument('--cases', default='', help='逗号分隔的病例号，默认取 test 里的全部病例')
    parser.add_argument('--organs', default='', help='逗号分隔的器官 id，默认全部')
    parser.add_argument('--out', type=Path, default=None, help='逐器官汇总 CSV')
    parser.add_argument('--show-saved', action='store_true',
                        help='读取 --out 指定的已有 CSV，按原始表格格式显示（不重新推理）')
    parser.add_argument('--ascii', action='store_true', help='用 +/- 代替 ±，兼容旧终端编码')
    parser.add_argument('--per-case-out', type=Path, default=None, help='逐病例明细 CSV')
    parser.add_argument('--raw-root', type=Path, default=RAW_ROOT)
    parser.add_argument('--slices-dir', type=Path, default=SLICES_DIR)
    parser.add_argument('--test-dir', '--data-dir', dest='test_dir', type=Path, default=TEST_DIR)
    parser.add_argument('--config', type=Path, default=ROOT / 'config' / 'nanseg.yaml',
                        help='模型配置（决定 in_channels / Crop_w / labels）。'
                             '默认 config/nanseg.yaml，若权重训练时用的 config 不同，请指定')
    parser.add_argument('--input-mode', choices=['auto', 'sliding', 'whole'], default='auto',
                        help='auto=按权重的 in_channels 自动选（1→sliding，>1→whole）；'
                             'sliding=整层铺滑窗（比如128裁剪训练的模型）；'
                             'whole=整层一次前向（整层进行训练）')
    parser.add_argument('--zdepth', type=int, default=None,
                        help='whole 模式的 z 邻层数，默认取权重的 in_channels（stage1 就是 3）')
    parser.add_argument('--factor', type=int, default=2,
                        help='whole 模式的输入降采样倍数（256² 训练填 2，128² 训练填 4）')
    parser.add_argument('--slices-source', choices=['dir', 'volume'], default='volume',
                        help='dir=读预处理切片目录；'
                             'volume=从原始 NRRD 重建完整体积（默认，覆盖全部层，'
                             '与 stage1_infer 同口径）')
    parser.add_argument('--cache-dir', type=Path, default=None, help='volume 模式的体数据缓存目录')
    parser.add_argument('--tile', type=int, default=128, help='滑窗边长（应等于训练 Crop_w）')
    parser.add_argument('--stride', type=int, default=96, help='滑窗步长（< tile 才有重叠）')
    parser.add_argument('--thr', type=float, default=0.5, help='概率二值化阈值')
    parser.add_argument('--batch', type=int, default=8, help='每次前向的窗口数')
    parser.add_argument('--no-skip-air', action='store_true',
                        help='不跳过纯空气窗口（更慢，用于对照）')
    parser.add_argument('--air-ratio', type=float, default=0.02,
                        help='窗口 CT 峰值 < 整层峰值 × 该比例 视为纯空气（只看 CT，不看 GT）')
    parser.add_argument('--hd95-mode', choices=['max', 'pooled'], default='max',
                        help="HD95：'max' 两方向 95 分位取最大（标准），'pooled' 为 MedPy 定义")
    parser.add_argument('--empty-policy', choices=['penalty', 'nan', 'skip'], default='penalty',
                        help='GT 非空而预测为空时：penalty 记体积对角线，nan 记 nan，skip 剔除该病例')
    args = parser.parse_args()
    load_config(args.config)

    cases = [c.strip() for c in args.cases.split(',') if c.strip()] or discover_cases(args.test_dir)
    organ_ids = ([int(x) for x in args.organs.split(',') if x.strip()]
                 if args.organs else list(range(N_ORGANS)))
    out_path = args.out or CHECKPOINT_DIR / f'metrics_3d_{args.ckpt.stem}.csv'
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
    unet, pff, epoch, in_ch, out_ch = build_models(args.ckpt, device)
    if args.input_mode == 'auto':
        args.input_mode = 'whole' if in_ch > 1 else 'sliding'
    if args.zdepth is None:
        args.zdepth = in_ch if in_ch > 1 else 3
    if out_ch != len(LABELS):
        print(f'警告：权重输出 {out_ch} 通道，但 labels 有 {len(LABELS)} 个，'
              f'评估只覆盖前 {min(out_ch, len(LABELS))} 个器官')
    print(f'checkpoint {args.ckpt.name}, epoch={epoch}, device={device}')
    print(f'从权重推断：in_channels={in_ch} -> input-mode={args.input_mode}, '
          f'zdepth={args.zdepth}, num_classes={out_ch}')
    print(f'config={Path(args.config).name} · labels={len(LABELS)}')
    print(f'cases={cases} hd95_mode={args.hd95_mode} empty_policy={args.empty_policy}')
    if args.input_mode == 'whole':
        print(f'推理口径：整层 · zdepth={args.zdepth} factor={args.factor} '
              f'batch={args.batch} thr={args.thr} ')
    else:
        print(f'推理口径：整层滑窗 tile={args.tile} stride={args.stride} batch={args.batch} '
              f'thr={args.thr} '
              f'{"（跳过纯空气窗口）" if not args.no_skip_air else "（不跳过任何窗口）"}')

    per_organ: OrderedDict = OrderedDict((LABELS[o], []) for o in organ_ids)
    layer_tot = {'tp': 0, 'fp': 0, 'fn': 0}
    records = []
    for case in cases:
        grid = CaseGrid(args.raw_root, case)
        get_slice, zs = make_slice_getter(case, args.slices_source, Path(args.slices_dir),
                                          args.raw_root, args.cache_dir)
        if not zs or len(set(zs)) != len(zs) or zs[0] < 0 or zs[-1] >= grid.n_slices:
            raise ValueError(f'{case}: invalid retained slice indices')
        missing = np.ones(grid.n_slices, dtype=bool)
        missing[zs] = False
        cover = f'（覆盖 {len(zs)}/{grid.n_slices} 层，来源={args.slices_source}）'
        print(f'{case}: {len(zs)} slices, z={zs[0]}..{zs[-1]}, '
              f'spacing={grid.spacing_xyz[0]:.3f}x{grid.spacing_xyz[1]:.3f}x{grid.spacing_xyz[2]:.2f} mm '
              + cover, flush=True)

        gt_volumes, spacings = {}, {}
        for organ_id in organ_ids:
            gt, spacing = grid.gt_volume(LABELS[organ_id])
            if gt is None:
                gt = np.zeros((grid.n_slices, SIZE, SIZE), dtype=bool)
            gt_volumes[organ_id] = gt
            spacings[organ_id] = spacing

        # 关键：对 zs 里所有层进行推理
        if args.input_mode == 'whole':
            pred_boxes = predict_case_whole(
                unet, pff, case, zs, device, get_slice, n_organs=N_ORGANS,
                zdepth=args.zdepth, factor=args.factor, thr=args.thr, batch=args.batch)
        else:
            pred_boxes = predict_case_sliding(
                unet, pff, case, zs, device, get_slice, n_organs=N_ORGANS,
                tile=args.tile, stride=args.stride, thr=args.thr, batch=args.batch,
                skip_air=not args.no_skip_air, air_ratio=args.air_ratio)

        for organ_id in organ_ids:
            gt_full = gt_volumes[organ_id]
            pred_full = np.zeros((grid.n_slices, SIZE, SIZE), dtype=bool)
            if len(zs):
                unpacked = np.unpackbits(pred_boxes[organ_id], axis=1)
                pred_full[zs] = unpacked.reshape(len(zs), SIZE, SIZE).astype(bool)
            omitted_gt = int(np.count_nonzero(gt_full[missing])) if missing.any() else 0
            metrics = binary_metrics(pred_full, gt_full, spacings[organ_id],
                                     hd95_mode=args.hd95_mode, empty_policy=args.empty_policy)

            # 层存在性混淆矩阵：现在预测覆盖每一层
            has_pred = pred_full.any(axis=(1, 2))
            has_gt = gt_full.any(axis=(1, 2))
            l_tp = int(np.count_nonzero(has_pred & has_gt))
            l_fp = int(np.count_nonzero(has_pred & ~has_gt))
            l_fn = int(np.count_nonzero(~has_pred & has_gt))
            layer_tot['tp'] += l_tp
            layer_tot['fp'] += l_fp
            layer_tot['fn'] += l_fn

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
                                       assd=metrics.assd,
                                       layer_tp=l_tp, layer_fp=l_fp, layer_fn=l_fn))
        del gt_volumes, pred_boxes

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
