from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from datagen.motion.action_space import (
    Act,
    ActionSpec,
    canonicalize,
    is_rotation_action,
)
from datagen.motion.pose_math import (
    compute_frames_for_action,
    integrate_multi_action,
    integrate_single_action,
)

DEFAULT_FOV_RAD = 0.9424777960769379  # 54 degrees


@dataclass
class Trajectory:
    name: str
    traj_id: str
    actions: list[ActionSpec]
    c2ws: np.ndarray
    Ks: np.ndarray
    key_frame_indices: list[int]
    params: dict[str, Any]
    actions_step: list[list[ActionSpec]]
    actions_prefix: list[list[ActionSpec]]


TRAJECTORY_REGISTRY: dict[str, Callable[..., Trajectory]] = {}


def register_trajectory(name: str) -> Callable[[Callable[..., Trajectory]], Callable[..., Trajectory]]:
    def _decorator(func: Callable[..., Trajectory]) -> Callable[..., Trajectory]:
        TRAJECTORY_REGISTRY[name] = func
        return func

    return _decorator


def _build_intrinsics(num_frames: int, image_wh: tuple[int, int]) -> np.ndarray:
    W, H = image_wh
    aspect = float(W) / float(H)
    if aspect >= 1.0:
        focal_x = 0.5 / np.tan(0.5 * DEFAULT_FOV_RAD)
        focal_y = focal_x * aspect
    else:
        focal_y = 0.5 / np.tan(0.5 * DEFAULT_FOV_RAD)
        focal_x = focal_y / aspect
    K_norm = np.array(
        [[focal_x, 0.0, 0.5], [0.0, focal_y, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    K = K_norm.copy()
    K[0, :] *= float(W)
    K[1, :] *= float(H)
    return np.repeat(K[None, :, :], num_frames, axis=0)


def _build_intrinsics_from_camera_cfg(
    num_frames: int,
    image_wh: tuple[int, int],
    camera_cfg: dict[str, Any] | None,
) -> np.ndarray:
    W, H = image_wh
    if not camera_cfg:
        return _build_intrinsics(num_frames=num_frames, image_wh=image_wh)

    fx_norm = camera_cfg.get("fx_norm")
    fy_norm = camera_cfg.get("fy_norm")
    cx_norm = camera_cfg.get("cx_norm")
    cy_norm = camera_cfg.get("cy_norm")
    if None not in (fx_norm, fy_norm, cx_norm, cy_norm):
        fx = float(fx_norm) * float(W)
        fy = float(fy_norm) * float(H)
        cx = float(cx_norm) * float(W)
        cy = float(cy_norm) * float(H)
    else:
        fov_deg = camera_cfg.get("fov_deg")
        if fov_deg is None:
            return _build_intrinsics(num_frames=num_frames, image_wh=image_wh)
        fov_axis = str(camera_cfg.get("fov_axis", "horizontal")).strip().lower()
        half_fov = np.deg2rad(float(fov_deg) * 0.5)
        tan_half = float(np.tan(half_fov))
        if tan_half <= 0:
            raise ValueError("camera.fov_deg must be within (0, 180)")
        if fov_axis == "horizontal":
            focal = 0.5 * float(W) / tan_half
        elif fov_axis == "vertical":
            focal = 0.5 * float(H) / tan_half
        elif fov_axis == "min":
            focal = 0.5 * float(min(W, H)) / tan_half
        else:
            raise ValueError("camera.fov_axis must be one of: horizontal, vertical, min")
        fx = focal
        fy = focal
        cx = 0.5 * float(W)
        cy = 0.5 * float(H)

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return np.repeat(K[None, :, :], num_frames, axis=0)


def _build_prefix(actions_step: list[list[ActionSpec]]) -> list[list[ActionSpec]]:
    prefix: list[list[ActionSpec]] = []
    acc: list[ActionSpec] = []
    for step in actions_step:
        acc = canonicalize(acc + step)
        prefix.append(acc.copy())
    return prefix


def _get_action_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    if "action" in cfg:
        return cfg["action"]
    return cfg


def _get_camera_cfg(cfg: dict[str, Any]) -> dict[str, Any] | None:
    camera_cfg = cfg.get("camera")
    if isinstance(camera_cfg, dict):
        return camera_cfg
    return None


def _max_steps_from_cfg(cfg: dict[str, Any]) -> tuple[float, float]:
    action_cfg = _get_action_cfg(cfg)
    distance_cfg = action_cfg["distance"]
    angle_cfg = action_cfg["angle"]
    return float(distance_cfg["max_step_per_frame"]), float(angle_cfg["max_step_per_frame"])


def _min_num_frames_from_cfg(cfg: dict[str, Any]) -> int | None:
    traj_cfg = cfg.get("trajectory", {})
    if not isinstance(traj_cfg, dict):
        return None
    raw = traj_cfg.get("min_num_frames")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return max(2, value)


def _camera_scale_from_cfg(cfg: dict[str, Any], magnitude: float) -> float:
    svc_cfg = cfg.get("svc", {})
    dynamic = bool(svc_cfg.get("dynamic_camera_scale", False))
    if not dynamic:
        return float(svc_cfg.get("camera_scale", 2.0))
    min_scale = float(svc_cfg.get("min_camera_scale", 0.5))
    return max(float(magnitude) * 0.5, min_scale)


def _single_step_trace(action: ActionSpec, max_step: float, num_frames: int) -> list[list[ActionSpec]]:
    step_mag = action.magnitude / max(1, num_frames - 1)
    per_frame = [[ActionSpec(action.act, step_mag)] for _ in range(num_frames - 1)]
    return [[]] + per_frame


def _build_single_step(
    name: str,
    act: Act,
    traj_id: str,
    image_wh: tuple[int, int],
    magnitude: float,
    cfg: dict[str, Any],
) -> Trajectory:
    action = ActionSpec(act=act, magnitude=float(magnitude))
    max_trans_step, max_yaw_step = _max_steps_from_cfg(cfg)
    min_num_frames = _min_num_frames_from_cfg(cfg)
    max_step = max_yaw_step if is_rotation_action(act) else max_trans_step
    c2ws, num_frames = integrate_single_action(
        action,
        max_step=max_step,
        min_num_frames=min_num_frames,
    )
    camera_cfg = _get_camera_cfg(cfg)
    Ks = _build_intrinsics_from_camera_cfg(num_frames=num_frames, image_wh=image_wh, camera_cfg=camera_cfg)
    camera_scale = _camera_scale_from_cfg(cfg, float(magnitude))
    key_frame_indices = [0, num_frames - 1]
    actions_step = _single_step_trace(action, max_step=max_step, num_frames=num_frames)
    actions_prefix = _build_prefix(actions_step)
    params = {
        "kind": "single_step",
        "action": act.value,
        "magnitude": float(magnitude),
        "num_frames": int(num_frames),
        "max_step_per_frame": float(max_step),
        "camera_scale": float(camera_scale),
    }
    return Trajectory(
        name=name,
        traj_id=traj_id,
        actions=[action],
        c2ws=c2ws,
        Ks=Ks,
        key_frame_indices=key_frame_indices,
        params=params,
        actions_step=actions_step,
        actions_prefix=actions_prefix,
    )


@register_trajectory("forward")
def make_forward(
    traj_id: str,
    image_wh: tuple[int, int],
    magnitude: float,
    cfg: dict[str, Any],
) -> Trajectory:
    return _build_single_step("forward", Act.FORWARD, traj_id, image_wh, magnitude, cfg)


@register_trajectory("backward")
def make_backward(
    traj_id: str,
    image_wh: tuple[int, int],
    magnitude: float,
    cfg: dict[str, Any],
) -> Trajectory:
    return _build_single_step("backward", Act.BACKWARD, traj_id, image_wh, magnitude, cfg)


@register_trajectory("turn_left")
def make_turn_left(
    traj_id: str,
    image_wh: tuple[int, int],
    magnitude: float,
    cfg: dict[str, Any],
) -> Trajectory:
    return _build_single_step("turn_left", Act.TURN_LEFT, traj_id, image_wh, magnitude, cfg)


@register_trajectory("turn_right")
def make_turn_right(
    traj_id: str,
    image_wh: tuple[int, int],
    magnitude: float,
    cfg: dict[str, Any],
) -> Trajectory:
    return _build_single_step("turn_right", Act.TURN_RIGHT, traj_id, image_wh, magnitude, cfg)


@register_trajectory("shift_left")
def make_shift_left(
    traj_id: str,
    image_wh: tuple[int, int],
    magnitude: float,
    cfg: dict[str, Any],
) -> Trajectory:
    return _build_single_step("shift_left", Act.SHIFT_LEFT, traj_id, image_wh, magnitude, cfg)


@register_trajectory("shift_right")
def make_shift_right(
    traj_id: str,
    image_wh: tuple[int, int],
    magnitude: float,
    cfg: dict[str, Any],
) -> Trajectory:
    return _build_single_step("shift_right", Act.SHIFT_RIGHT, traj_id, image_wh, magnitude, cfg)


@register_trajectory("multi_step")
def make_multi_step(
    traj_id: str,
    image_wh: tuple[int, int],
    action_specs: list[ActionSpec],
    cfg: dict[str, Any],
) -> Trajectory:
    if not action_specs:
        raise ValueError("action_specs must not be empty for multi_step trajectory")

    max_trans_step, max_yaw_step = _max_steps_from_cfg(cfg)
    min_num_frames = _min_num_frames_from_cfg(cfg)
    c2ws, key_frame_indices = integrate_multi_action(
        actions=action_specs,
        max_trans_step=max_trans_step,
        max_yaw_step=max_yaw_step,
        min_num_frames=min_num_frames,
    )
    camera_cfg = _get_camera_cfg(cfg)
    Ks = _build_intrinsics_from_camera_cfg(
        num_frames=int(c2ws.shape[0]),
        image_wh=image_wh,
        camera_cfg=camera_cfg,
    )
    total_magnitude = float(sum(float(a.magnitude) for a in action_specs))
    camera_scale = _camera_scale_from_cfg(cfg, total_magnitude)

    actions_step: list[list[ActionSpec]] = [[]]
    for action in action_specs:
        max_step = max_yaw_step if is_rotation_action(action.act) else max_trans_step
        num_frames = compute_frames_for_action(
            action.magnitude,
            max_step,
            min_num_frames=min_num_frames,
        )
        step_mag = action.magnitude / max(1, num_frames - 1)
        actions_step.extend([[ActionSpec(action.act, step_mag)] for _ in range(num_frames - 1)])
    actions_prefix = _build_prefix(actions_step)

    if len(actions_step) != int(c2ws.shape[0]):
        raise RuntimeError("actions_step and c2ws length mismatch in make_multi_step")

    params = {
        "kind": "multi_step",
        "num_actions": len(action_specs),
        "num_frames": int(c2ws.shape[0]),
        "key_frame_indices": [int(v) for v in key_frame_indices],
        "actions": [{"act": a.act.value, "magnitude": float(a.magnitude)} for a in action_specs],
        "max_trans_step_per_frame": float(max_trans_step),
        "max_yaw_step_per_frame": float(max_yaw_step),
        "camera_scale": float(camera_scale),
    }
    return Trajectory(
        name="multi_step",
        traj_id=traj_id,
        actions=action_specs,
        c2ws=c2ws,
        Ks=Ks,
        key_frame_indices=key_frame_indices,
        params=params,
        actions_step=actions_step,
        actions_prefix=actions_prefix,
    )


def build_trajectory(name: str, **kwargs: Any) -> Trajectory:
    if name not in TRAJECTORY_REGISTRY:
        raise KeyError(f"Unknown trajectory name: {name}")
    return TRAJECTORY_REGISTRY[name](**kwargs)

