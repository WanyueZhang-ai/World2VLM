from __future__ import annotations

from typing import Any


def frame_record(
    uid: str,
    scene_id: str,
    anchor_path: str,
    image_path: str,
    traj_id: str,
    traj_params: dict[str, Any],
    frame_idx: int,
    actions_prefix: list[str],
    actions_step: list[str],
    svc_options: dict[str, Any],
    qc: dict[str, Any],
) -> dict[str, Any]:
    # Keep `svc_options` field name for compatibility with existing prompt tools.
    return {
        "uid": uid,
        "scene_id": scene_id,
        "anchor_path": anchor_path,
        "image_path": image_path,
        "traj_id": traj_id,
        "traj_params": traj_params,
        "frame_idx": frame_idx,
        "actions_prefix": actions_prefix,
        "actions_step": actions_step,
        "svc_options": svc_options,
        "qc": qc,
    }

