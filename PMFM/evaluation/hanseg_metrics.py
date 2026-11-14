
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage

METRIC_KEYS = ('dice', 'jaccard', 'hd95', 'assd')
SUMMARY_FIELDS = ['n'] + [f'{k}_{stat}' for k in METRIC_KEYS for stat in ('mean', 'std')]


BILATERAL_GROUPS = OrderedDict([
    ('A_Carotid', ('A_Carotid_L', 'A_Carotid_R')),
    ('Cochlea', ('Cochlea_L', 'Cochlea_R')),
    ('Eye_A', ('Eye_AL', 'Eye_AR')),
    ('Eye_P', ('Eye_PL', 'Eye_PR')),
    ('Glnd_Lacrimal', ('Glnd_Lacrimal_L', 'Glnd_Lacrimal_R')),
    ('Glnd_Submands', ('Glnd_Submand_L', 'Glnd_Submand_R')),
    ('OpticNrv', ('OpticNrv_L', 'OpticNrv_R')),
    ('Parotids', ('Parotid_L', 'Parotid_R')),
])

STATUS_VALID = 'valid'
STATUS_GT_EMPTY = 'gt_empty'
STATUS_BOTH_EMPTY = 'both_empty'
STATUS_EMPTY_PREDICTION = 'empty_prediction'


@dataclass(frozen=True)
class CaseMetrics:


    


    dice: float
    jaccard: float
    hd95: float
    assd: float
    status: str
    evaluated: bool
    hd95_pooled: float = float('nan')

    def as_list(self) -> list[float]:
        return [self.dice, self.jaccard, self.hd95, self.assd]


def _as_bool(mask) -> np.ndarray:
    return np.asarray(mask).astype(bool)


def physical_diagonal(shape: Iterable[int], spacing: Sequence[float]) -> float:

    extent = (np.asarray(tuple(shape), dtype=np.float64) - 1.0) * np.asarray(spacing, dtype=np.float64)
    return float(np.linalg.norm(extent))


def dice_coefficient(pred, gt) -> float:
    pred, gt = _as_bool(pred), _as_bool(gt)
    total = int(pred.sum()) + int(gt.sum())
    if total == 0:
        return float('nan')
    return float(2.0 * np.logical_and(pred, gt).sum() / total)


def jaccard_coefficient(pred, gt) -> float:
    pred, gt = _as_bool(pred), _as_bool(gt)
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return float('nan')
    return float(np.logical_and(pred, gt).sum() / union)


def _surface(mask: np.ndarray) -> np.ndarray:
    footprint = ndimage.generate_binary_structure(mask.ndim, 1)
    return mask & ~ndimage.binary_erosion(mask, structure=footprint, iterations=1, border_value=0)


def _bounding_box(union: np.ndarray, pad: int = 2) -> tuple[slice, ...]:

    coords = np.argwhere(union)
    lo = np.maximum(coords.min(axis=0) - pad, 0)
    hi = np.minimum(coords.max(axis=0) + pad + 1, union.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _directed_surface_distances(source: np.ndarray, target: np.ndarray,
                                spacing: Sequence[float]) -> np.ndarray:

    target_surface = _surface(target)
    source_surface = _surface(source)
    distance_map = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    return distance_map[source_surface]


def surface_distances(pred, gt, spacing: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:

    pred, gt = _as_bool(pred), _as_bool(gt)
    if not pred.any() or not gt.any():
        raise ValueError('surface_distances 需要两个掩膜都非空')
    box = _bounding_box(pred | gt)
    p, g = pred[box], gt[box]
    return (_directed_surface_distances(p, g, spacing),
            _directed_surface_distances(g, p, spacing))


def hd95_from_distances(d12: np.ndarray, d21: np.ndarray, mode: str = 'max') -> float:

    if mode == 'max':
        return float(max(np.percentile(d12, 95), np.percentile(d21, 95)))
    if mode == 'pooled':
        return float(np.percentile(np.concatenate([d12, d21]), 95))
    raise ValueError(f"未知 hd95_mode: {mode}（可选 'max' / 'pooled'）")


def assd_from_distances(d12: np.ndarray, d21: np.ndarray) -> float:

    return float(np.concatenate([d12, d21]).mean())


def binary_metrics(pred, gt, spacing: Sequence[float], *,
                   hd95_mode: str = 'max',
                   empty_policy: str = 'penalty') -> CaseMetrics:

    pred, gt = _as_bool(pred), _as_bool(gt)
    spacing = tuple(float(s) for s in spacing)
    gt_any, pred_any = bool(gt.any()), bool(pred.any())

    if not gt_any:
        status = STATUS_BOTH_EMPTY if not pred_any else STATUS_GT_EMPTY
        return CaseMetrics(float('nan'), float('nan'), float('nan'), float('nan'), status, False)

    if not pred_any:
        if empty_policy == 'skip':
            return CaseMetrics(0.0, 0.0, float('nan'), float('nan'),
                               STATUS_EMPTY_PREDICTION, False)
        if empty_policy == 'nan':
            return CaseMetrics(0.0, 0.0, float('nan'), float('nan'),
                               STATUS_EMPTY_PREDICTION, True)
        if empty_policy != 'penalty':
            raise ValueError(f"未知 empty_policy: {empty_policy}（可选 'penalty' / 'nan' / 'skip'）")
        penalty = physical_diagonal(gt.shape, spacing)
        return CaseMetrics(0.0, 0.0, penalty, penalty, STATUS_EMPTY_PREDICTION, True)

    d12, d21 = surface_distances(pred, gt, spacing)
    return CaseMetrics(dice_coefficient(pred, gt),
                       jaccard_coefficient(pred, gt),
                       hd95_from_distances(d12, d21, hd95_mode),
                       assd_from_distances(d12, d21),
                       STATUS_VALID, True,
                       hd95_pooled=hd95_from_distances(d12, d21, 'pooled'))


def summarize(cases: Sequence[CaseMetrics], ddof: int = 1) -> OrderedDict:

    vals = np.asarray([c.as_list() for c in cases if c.evaluated], dtype=np.float64)
    if vals.size == 0:
        vals = np.zeros((0, len(METRIC_KEYS)))
    n = int(vals.shape[0])
    row: OrderedDict[str, float] = OrderedDict(n=n)
    if n == 0:
        for key in METRIC_KEYS:
            row[f'{key}_mean'] = float('nan')
            row[f'{key}_std'] = float('nan')
        return row
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(vals, axis=0)
        std = np.nanstd(vals, axis=0, ddof=ddof) if n > ddof else np.full(len(METRIC_KEYS), np.nan)
    for i, key in enumerate(METRIC_KEYS):
        row[f'{key}_mean'] = float(mean[i])
        row[f'{key}_std'] = float(std[i])
    return row


def merge_bilateral(per_organ: Mapping[str, Sequence[CaseMetrics]]) -> dict[str, list[CaseMetrics]]:

    merged = {}
    for name, (left, right) in BILATERAL_GROUPS.items():
        if left in per_organ and right in per_organ:
            merged[name] = list(per_organ[left]) + list(per_organ[right])
    return merged


def format_mean_std(mean: float, std: float, digits: int = 2) -> str:

    if mean is None or not np.isfinite(mean):
        return 'n/a'
    if std is None or not np.isfinite(std):
        return f'{mean:.{digits}f} ± n/a'
    return f'{mean:.{digits}f} ± {std:.{digits}f}'
