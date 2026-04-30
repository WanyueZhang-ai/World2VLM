from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from seva.geometry import DEFAULT_FOV_RAD, get_default_intrinsics

from svc_dataset_gen.traj.action_space import Action, Act, canonicalize
from svc_dataset_gen.traj.pose_math import (
    actions_from_pose_delta,
    integrate_action_sequence,
)


@dataclass
class Trajectory:
    name: str
    traj_id: str
    params: dict
    c2ws: np.ndarray
    Ks: np.ndarray
    actions_step: list[list[Action]]
    actions_prefix: list[list[Action]]


def _build_intrinsics(num_frames: int, image_wh: tuple[int, int]) -> np.ndarray:
    W, H = image_wh
    fovs = torch.full((num_frames,), float(DEFAULT_FOV_RAD))
    Ks = get_default_intrinsics(fovs, aspect_ratio=W / H)
    Ks[:, :2] *= torch.tensor([W, H]).reshape(1, -1, 1).repeat(num_frames, 1, 1)
    return Ks.numpy()


def _build_prefix(actions_step: list[list[Action]]) -> list[list[Action]]:
    prefix = []
    acc: list[Action] = []
    for step in actions_step:
        acc = canonicalize(acc + step)
        prefix.append(acc.copy())
    return prefix


def make_T1(
    traj_id: str,
    num_frames: int,
    image_wh: tuple[int, int],
    yaw_step_deg: float,
    direction: int,
) -> Trajectory:
    action = Action(Act.TR if direction > 0 else Act.TL, yaw_step_deg)
    actions_step = [[]] + [[action] for _ in range(num_frames - 1)]
    c2ws = integrate_action_sequence(actions_step)
    Ks = _build_intrinsics(num_frames, image_wh)
    actions_prefix = _build_prefix(actions_step)
    total_yaw = yaw_step_deg * max(0, num_frames - 1)
    params = {
        "num_frames": num_frames,
        "yaw_step_deg": yaw_step_deg,
        "yaw_total_deg": total_yaw,
    }
    return Trajectory("T1", traj_id, params, c2ws, Ks, actions_step, actions_prefix)


def make_T2(
    traj_id: str,
    num_frames: int,
    image_wh: tuple[int, int],
    trans_step_m: float,
    direction: int,
) -> Trajectory:
    action = Action(Act.RIGHT if direction > 0 else Act.LEFT, trans_step_m)
    actions_step = [[]] + [[action] for _ in range(num_frames - 1)]
    c2ws = integrate_action_sequence(actions_step)
    Ks = _build_intrinsics(num_frames, image_wh)
    actions_prefix = _build_prefix(actions_step)
    total_trans = trans_step_m * max(0, num_frames - 1)
    params = {
        "num_frames": num_frames,
        "trans_step_m": trans_step_m,
        "trans_total_m": total_trans,
    }
    return Trajectory("T2", traj_id, params, c2ws, Ks, actions_step, actions_prefix)


def make_T3(
    traj_id: str,
    num_frames: int,
    image_wh: tuple[int, int],
    trans_step_m: float,
    direction: int,
) -> Trajectory:
    action = Action(Act.FWD if direction > 0 else Act.BACK, trans_step_m)
    actions_step = [[]] + [[action] for _ in range(num_frames - 1)]
    c2ws = integrate_action_sequence(actions_step)
    Ks = _build_intrinsics(num_frames, image_wh)
    actions_prefix = _build_prefix(actions_step)
    total_trans = trans_step_m * max(0, num_frames - 1)
    params = {
        "num_frames": num_frames,
        "trans_step_m": trans_step_m,
        "trans_total_m": total_trans,
    }
    return Trajectory("T3", traj_id, params, c2ws, Ks, actions_step, actions_prefix)


def make_T4(
    traj_id: str,
    image_wh: tuple[int, int],
    turn_steps: list[float],
    move_steps: list[float],
    rng: np.random.Generator,
    num_steps: int = 3,
    action_substeps: int = 1,
) -> Trajectory:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive for T4.")
    if action_substeps <= 0:
        raise ValueError("action_substeps must be positive for T4.")
    if not turn_steps or not move_steps:
        raise ValueError("turn_steps and move_steps must be non-empty for T4.")

    opposite = {
        Act.TL: Act.TR,
        Act.TR: Act.TL,
        Act.FWD: Act.BACK,
        Act.BACK: Act.FWD,
        Act.LEFT: Act.RIGHT,
        Act.RIGHT: Act.LEFT,
    }
    candidates = [Act.TL, Act.TR, Act.FWD, Act.BACK, Act.LEFT, Act.RIGHT]

    actions_step: list[list[Action]] = [[]]
    prev_act: Act | None = None
    for _ in range(num_steps):
        choices = [
            act
            for act in candidates
            if prev_act is None or (act != prev_act and opposite[act] != prev_act)
        ]
        act = rng.choice(np.array(choices, dtype=object))
        if act in (Act.TL, Act.TR):
            step = float(rng.choice(np.array(turn_steps, dtype=np.float32)))
        else:
            step = float(rng.choice(np.array(move_steps, dtype=np.float32)))
        step = step / float(action_substeps)
        for _ in range(action_substeps):
            actions_step.append([Action(act, step)])
        prev_act = act
    num_frames = len(actions_step)
    c2ws = integrate_action_sequence(actions_step)
    Ks = _build_intrinsics(num_frames, image_wh)
    actions_prefix = _build_prefix(actions_step)
    params = {
        "num_frames": num_frames,
        "turn_steps": [float(v) for v in turn_steps],
        "move_steps": [float(v) for v in move_steps],
        "num_steps": num_steps,
        "action_substeps": action_substeps,
    }
    return Trajectory("T4", traj_id, params, c2ws, Ks, actions_step, actions_prefix)


def _look_at_c2w(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - position
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward], axis=1)
    return np.concatenate([R, position.reshape(3, 1)], axis=1)


def make_T5_orbit(
    traj_id: str,
    num_frames: int,
    image_wh: tuple[int, int],
    pivot: np.ndarray,
    radius_m: float,
    delta_deg: float,
    yaw_step_deg: float,
    trans_step_m: float,
) -> Trajectory:
    thetas = np.linspace(0.0, np.deg2rad(delta_deg), num_frames, endpoint=True)
    c2ws = []
    for theta in thetas:
        pos = pivot + np.array(
            [radius_m * np.sin(theta), 0.0, radius_m * np.cos(theta)],
            dtype=np.float32,
        )
        c2ws.append(_look_at_c2w(pos, pivot))
    c2ws = np.stack(c2ws, axis=0).astype(np.float32)
    Ks = _build_intrinsics(num_frames, image_wh)

    actions_step: list[list[Action]] = [[]]
    for idx in range(1, num_frames):
        step_actions = actions_from_pose_delta(
            c2ws[idx - 1],
            c2ws[idx],
            yaw_step_deg=yaw_step_deg,
            trans_step_m=trans_step_m,
        )
        actions_step.append(canonicalize(step_actions))
    actions_prefix = _build_prefix(actions_step)
    params = {
        "num_frames": num_frames,
        "delta_deg": delta_deg,
        "radius_m": radius_m,
        "pivot_depth_m": float(pivot[2]),
    }
    return Trajectory("T5", traj_id, params, c2ws, Ks, actions_step, actions_prefix)


def pivot_from_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    K: np.ndarray,
    c2w: np.ndarray,
    depth_m: float,
) -> np.ndarray:
    u0, v0, u1, v1 = bbox_xyxy
    u = 0.5 * (u0 + u1)
    v = 0.5 * (v0 + v1)
    K_inv = np.linalg.inv(K)
    ray = K_inv @ np.array([u, v, 1.0], dtype=np.float32)
    ray = ray / (np.linalg.norm(ray) + 1e-8)
    p_cam = ray * depth_m
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    p_world = R @ p_cam + t
    return p_world.astype(np.float32)
