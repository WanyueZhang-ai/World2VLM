from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datagen.motion.action_space import ActionSpec, actions_to_tokens, canonicalize


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected dict JSON at {path}, got {type(data)}")
    return data


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=True) + "\n")


def prepare_scene_dir(out_root: Path, scene_id: str) -> Path:
    scene_dir = out_root / "scenes" / scene_id
    ensure_dir(scene_dir)
    return scene_dir


def prepare_traj_dir(scene_dir: Path, traj_id: str) -> Path:
    traj_dir = scene_dir / f"traj_{traj_id}"
    ensure_dir(traj_dir)
    return traj_dir


def merge_actions_between(
    actions_step: list[list[ActionSpec]],
    start_idx: int,
    end_idx: int,
) -> list[ActionSpec]:
    merged: list[ActionSpec] = []
    for i in range(start_idx + 1, end_idx + 1):
        merged.extend(actions_step[i])
    return canonicalize(merged)


def write_traj_bundle(
    traj_dir: Path,
    traj_params: dict[str, Any],
    actions_step: list[list[ActionSpec]],
    actions_prefix: list[list[ActionSpec]],
    frame_ids_kept: list[int],
    transforms: dict[str, Any] | None,
    quality_scores: dict[str, Any] | None = None,
    key_frame_indices: list[int] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "traj_params": traj_params,
        "frame_ids_kept": frame_ids_kept,
        "actions_step": [actions_to_tokens(a) for a in actions_step],
        "actions_prefix": [actions_to_tokens(a) for a in actions_prefix],
        "transforms": transforms,
    }
    if quality_scores is not None:
        payload["quality_scores"] = quality_scores
    if key_frame_indices is not None:
        payload["key_frame_indices"] = key_frame_indices
    write_json(traj_dir / "traj.json", payload)

