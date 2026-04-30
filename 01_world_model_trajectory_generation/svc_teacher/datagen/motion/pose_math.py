from __future__ import annotations

import math

import numpy as np

from datagen.motion.action_space import Act, ActionSpec, is_rotation_action


def rot_yaw_deg(yaw_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def rt_to_c2w(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.concatenate([R, t.reshape(3, 1)], axis=1)


def c2w_to_mat4(c2w: np.ndarray) -> np.ndarray:
    if c2w.shape == (4, 4):
        return c2w
    if c2w.shape != (3, 4):
        raise ValueError(f"c2w shape must be (3,4) or (4,4), got {c2w.shape}")
    bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=c2w.dtype)
    return np.concatenate([c2w, bottom], axis=0)


def yaw_from_rot(R: np.ndarray) -> float:
    return math.degrees(math.atan2(float(R[0, 2]), float(R[2, 2])))


def compute_frames_for_action(
    magnitude: float,
    max_step: float,
    min_num_frames: int | None = None,
) -> int:
    """Derive frame count from target magnitude and max per-frame step."""
    if max_step <= 0:
        raise ValueError("max_step must be positive")
    # Use ceil to guarantee per-frame step does not exceed max_step.
    ratio = abs(magnitude) / max_step
    n_steps = max(1, int(math.ceil(ratio - 1e-9)))
    frames = n_steps + 1
    if min_num_frames is not None:
        frames = max(frames, int(min_num_frames))
    return frames


def _apply_single_step(
    action: ActionSpec,
    step_per_frame: float,
    yaw_deg: float,
    t: np.ndarray,
) -> tuple[float, np.ndarray]:
    if action.act == Act.TURN_LEFT:
        yaw_deg -= step_per_frame
        return yaw_deg, t
    if action.act == Act.TURN_RIGHT:
        yaw_deg += step_per_frame
        return yaw_deg, t

    cam_delta = np.zeros(3, dtype=np.float32)
    if action.act == Act.FORWARD:
        cam_delta[2] += step_per_frame
    elif action.act == Act.BACKWARD:
        cam_delta[2] -= step_per_frame
    elif action.act == Act.SHIFT_LEFT:
        cam_delta[0] -= step_per_frame
    elif action.act == Act.SHIFT_RIGHT:
        cam_delta[0] += step_per_frame
    else:
        raise ValueError(f"Unsupported action {action.act}")

    R = rot_yaw_deg(yaw_deg)
    t = t + R @ cam_delta
    return yaw_deg, t


def integrate_single_action(
    action: ActionSpec,
    max_step: float,
    min_num_frames: int | None = None,
    yaw0: float = 0.0,
    t0: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Integrate one action into a [N,3,4] c2w sequence."""
    if t0 is None:
        t = np.zeros(3, dtype=np.float32)
    else:
        t = np.asarray(t0, dtype=np.float32).copy()

    num_frames = compute_frames_for_action(action.magnitude, max_step, min_num_frames=min_num_frames)
    step_per_frame = action.magnitude / max(1, num_frames - 1)
    yaw_deg = float(yaw0)

    c2ws = [rt_to_c2w(rot_yaw_deg(yaw_deg), t.copy())]
    for _ in range(num_frames - 1):
        yaw_deg, t = _apply_single_step(action, step_per_frame, yaw_deg, t)
        c2ws.append(rt_to_c2w(rot_yaw_deg(yaw_deg), t.copy()))
    return np.stack(c2ws, axis=0), num_frames


def integrate_multi_action(
    actions: list[ActionSpec],
    max_trans_step: float,
    max_yaw_step: float,
    min_num_frames: int | None = None,
) -> tuple[np.ndarray, list[int]]:
    """
    Integrate multi-step actions into one continuous c2w sequence.

    Returns:
        c2ws: [N,3,4]
        key_frame_indices: indices at each action boundary (including 0)
    """
    yaw_deg = 0.0
    t = np.zeros(3, dtype=np.float32)
    c2ws_all = [rt_to_c2w(rot_yaw_deg(yaw_deg), t.copy())]
    key_frame_indices = [0]

    for action in actions:
        max_step = max_yaw_step if is_rotation_action(action.act) else max_trans_step
        seg_c2ws, _ = integrate_single_action(
            action,
            max_step,
            min_num_frames=min_num_frames,
            yaw0=yaw_deg,
            t0=t,
        )
        if seg_c2ws.shape[0] > 1:
            c2ws_all.extend(seg_c2ws[1:])

        last = seg_c2ws[-1]
        yaw_deg = yaw_from_rot(last[:3, :3])
        t = np.asarray(last[:3, 3], dtype=np.float32)
        key_frame_indices.append(len(c2ws_all) - 1)

    return np.stack(c2ws_all, axis=0), key_frame_indices

