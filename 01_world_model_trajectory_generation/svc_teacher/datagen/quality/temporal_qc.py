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

