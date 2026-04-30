from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

DISTANCE_GRID = 0.1
ANGLE_GRID_DEG = 10.0
DISTANCE_TOL = 0.01
ANGLE_TOL_DEG = 1.0
SAMPLES_DIRNAME = "samples-rgb"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_scene_dir(output_root: Path, scene_id: str) -> Path:
    return ensure_dir(output_root / "scenes" / scene_id)


def prepare_traj_dir(scene_dir: Path, traj_id: str) -> Path:
    traj_dir = ensure_dir(scene_dir / f"traj_{traj_id}")
    ensure_dir(traj_dir / SAMPLES_DIRNAME)
    return traj_dir


def _quantize_half_up(value: float, grid: float) -> float:
    if grid <= 0:
        raise ValueError("grid must be positive")
    steps = math.floor((abs(float(value)) / grid) + 0.5 + 1e-12)
    return round(steps * grid, 6)


def _is_close_to_grid(value: float, grid: float, tol: float) -> bool:
    magnitude = abs(float(value))
    if magnitude <= tol:
        return False
    quantized = _quantize_half_up(magnitude, grid)
    return abs(quantized - magnitude) <= tol


def _relative_path(path: Path | str | None, base_dir: Path) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        return candidate.relative_to(base_dir).as_posix()
    except ValueError:
        return str(candidate)


def _jsonify(value: Any, *, base_dir: Path | None = None) -> Any:
    if is_dataclass(value):
        return _jsonify(asdict(value), base_dir=base_dir)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonify(inner, base_dir=base_dir)
            for key, inner in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonify(item, base_dir=base_dir) for item in value]
    if isinstance(value, Path):
        if base_dir is not None:
            return _relative_path(value, base_dir)
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    return value


def select_sample_indices(
    progress_points: Sequence[Mapping[str, Any]],
    distance_grid: float = DISTANCE_GRID,
    angle_grid: float = ANGLE_GRID_DEG,
    distance_tol: float = DISTANCE_TOL,
    angle_tol: float = ANGLE_TOL_DEG,
) -> list[int]:
    if not progress_points:
        return []

    selected: list[int] = []
    for index, point in enumerate(progress_points):
        frame_index = int(point["frame_index"])
        metric_kind = str(point.get("metric_kind", "translation"))
        metric_value = float(point.get("metric_value", 0.0))
        is_action_end = bool(point.get("is_action_end", False))
        is_first = index == 0
        is_last = index == len(progress_points) - 1

        keep = is_first or is_last or is_action_end
        if not keep:
            if metric_kind == "translation":
                keep = _is_close_to_grid(metric_value, distance_grid, distance_tol)
            elif metric_kind == "rotation":
                keep = _is_close_to_grid(metric_value, angle_grid, angle_tol)

        if keep and frame_index not in selected:
            selected.append(frame_index)

    return selected


def _normalize_video_tensor(video_tensor: torch.Tensor) -> torch.Tensor:
    if video_tensor.ndim == 5:
        if video_tensor.shape[0] != 1:
            raise ValueError("expected batch size 1 when saving selected frames")
        video_tensor = video_tensor[0]
    if video_tensor.ndim != 4:
        raise ValueError("video tensor must have shape [C, F, H, W] or [1, C, F, H, W]")
    return video_tensor.detach().cpu()


def save_selected_frames(
    video_tensor: torch.Tensor,
    frame_indices: Sequence[int],
    output_dir: Path,
) -> list[Path]:
    video = _normalize_video_tensor(video_tensor)
    samples_dir = ensure_dir(output_dir / SAMPLES_DIRNAME)
    max_frames = int(video.shape[1])

    saved_paths: list[Path] = []
    for frame_index in frame_indices:
        frame_idx = int(frame_index)
        if frame_idx < 0 or frame_idx >= max_frames:
            raise IndexError(f"frame index {frame_idx} out of range for {max_frames} frames")
        frame = video[:, frame_idx].permute(1, 2, 0)
        frame_uint8 = (frame * 255.0).clamp(0, 255).to(torch.uint8).numpy()
        path = samples_dir / f"frame_{frame_idx:04d}.png"
        Image.fromarray(frame_uint8).save(path)
        saved_paths.append(path)

    return saved_paths


def _rounded_motion_summary(
    point: Mapping[str, Any],
    distance_grid: float = DISTANCE_GRID,
    angle_grid: float = ANGLE_GRID_DEG,
) -> dict[str, Any]:
    metric_kind = str(point.get("metric_kind", "translation"))
    metric_value = float(point.get("metric_value", 0.0))
    rounded_grid = distance_grid if metric_kind == "translation" else angle_grid
    summary = {
        "frame_index": int(point["frame_index"]),
        "metric_kind": metric_kind,
        "metric_value": round(metric_value, 6),
        "rounded_metric_value": _quantize_half_up(metric_value, rounded_grid),
        "distance_m": round(float(point.get("distance_m", 0.0)), 6),
        "angle_deg": round(float(point.get("angle_deg", 0.0)), 6),
        "is_action_end": bool(point.get("is_action_end", False)),
    }
    if "action_name" in point:
        summary["action_name"] = str(point["action_name"])
    if "action_index" in point:
        summary["action_index"] = int(point["action_index"])
    return summary


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonify(payload, base_dir=path.parent),
            handle,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return path


def write_traj_json(
    traj_dir: Path,
    *,
    prompt: str,
    anchor_path: str | Path,
    actions: Sequence[Any],
    latent_chunks: Sequence[Any],
    video_length: int,
    retained_frame_indices: Sequence[int],
    progress_points: Sequence[Mapping[str, Any]],
    sample_paths: Sequence[Path] | None = None,
    pose_path: Path | None = None,
    mp4_path: Path | None = None,
) -> Path:
    retained_set = {int(frame_index) for frame_index in retained_frame_indices}
    retained_motion_summaries = [
        _rounded_motion_summary(point)
        for point in progress_points
        if int(point["frame_index"]) in retained_set
    ]
    payload = {
        "prompt": prompt,
        "anchor_path": str(anchor_path),
        "actions": _jsonify(actions, base_dir=traj_dir),
        "latent_chunks": _jsonify(latent_chunks, base_dir=traj_dir),
        "video_length": int(video_length),
        "retained_frame_indices": [int(frame_index) for frame_index in retained_frame_indices],
        "retained_motion_summaries": retained_motion_summaries,
        "sample_paths": _jsonify(sample_paths or [], base_dir=traj_dir),
        "pose_path": _relative_path(pose_path, traj_dir),
        "mp4_path": _relative_path(mp4_path, traj_dir),
    }
    return write_json(traj_dir / "traj.json", payload)


def append_manifest_record(path: Path, record: Mapping[str, Any]) -> Path:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonify(record), ensure_ascii=True, sort_keys=True))
        handle.write("\n")
    return path


def write_stats(path: Path, stats: Mapping[str, Any]) -> Path:
    return write_json(path, stats)
