from __future__ import annotations

import numpy as np


def uniform_sample_indices(
    num_frames: int,
    keep_k: int,
    include_first: bool = False,
    include_last: bool = True,
) -> list[int]:
    if num_frames <= 0:
        return []
    if keep_k >= num_frames:
        return list(range(num_frames))
    idxs = np.linspace(0, num_frames - 1, keep_k).round().astype(int).tolist()
    if include_first and 0 not in idxs:
        idxs[0] = 0
    if include_last and (num_frames - 1) not in idxs:
        idxs[-1] = num_frames - 1
    idxs = sorted(set(idxs))
    return idxs


def select_frame_indices(
    num_frames: int,
    keep_k: int,
    is_orbit: bool = False,
) -> list[int]:
    if is_orbit:
        return uniform_sample_indices(num_frames, keep_k, include_last=True)
    return uniform_sample_indices(num_frames, keep_k, include_last=True)
