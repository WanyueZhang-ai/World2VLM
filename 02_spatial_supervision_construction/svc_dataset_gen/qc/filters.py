from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _frame_diff(a: Path, b: Path) -> float:
    img_a = np.array(Image.open(a).convert("RGB"), dtype=np.float32) / 255.0
    img_b = np.array(Image.open(b).convert("RGB"), dtype=np.float32) / 255.0
    return float(np.mean(np.abs(img_a - img_b)))


def check_temporal_smoothness(
    frame_paths: list[Path],
    min_diff: float = 0.01,
    max_diff: float = 0.4,
) -> tuple[bool, dict]:
    if len(frame_paths) < 2:
        return True, {"min_diff": None, "max_diff": None}
    diffs = [_frame_diff(frame_paths[i - 1], frame_paths[i]) for i in range(1, len(frame_paths))]
    min_v = float(np.min(diffs))
    max_v = float(np.max(diffs))
    passed = min_v >= min_diff and max_v <= max_diff
    return passed, {"min_diff": min_v, "max_diff": max_v}


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def check_bbox_track_stability(
    bboxes_xyxy: list[tuple[float, float, float, float]],
    max_center_shift: float = 30.0,
    min_iou: float = 0.2,
) -> tuple[bool, dict]:
    if len(bboxes_xyxy) < 2:
        return True, {"max_center_shift": None, "min_iou": None}
    centers = [((x0 + x1) * 0.5, (y0 + y1) * 0.5) for x0, y0, x1, y1 in bboxes_xyxy]
    shifts = [
        float(np.hypot(centers[i][0] - centers[i - 1][0], centers[i][1] - centers[i - 1][1]))
        for i in range(1, len(centers))
    ]
    ious = [
        _iou_xyxy(bboxes_xyxy[i - 1], bboxes_xyxy[i]) for i in range(1, len(bboxes_xyxy))
    ]
    max_shift = float(np.max(shifts))
    min_iou_val = float(np.min(ious))
    passed = max_shift <= max_center_shift and min_iou_val >= min_iou
    return passed, {"max_center_shift": max_shift, "min_iou": min_iou_val}
