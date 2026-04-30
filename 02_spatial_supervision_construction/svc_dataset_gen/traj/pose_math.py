from __future__ import annotations

import math

import numpy as np

from svc_dataset_gen.traj.action_space import Act, Action, quantize_actions


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
    bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=c2w.dtype)
    return np.concatenate([c2w, bottom], axis=0)


def integrate_action_sequence(
    actions_step: list[list[Action]],
    yaw0_deg: float = 0.0,
    t0: np.ndarray | None = None,
) -> np.ndarray:
    if t0 is None:
        t = np.zeros(3, dtype=np.float32)
    else:
        t = np.array(t0, dtype=np.float32)
    yaw = yaw0_deg
    c2ws = []
    R = rot_yaw_deg(yaw)
    c2ws.append(rt_to_c2w(R, t))
    for step_actions in actions_step[1:]:
        for action in step_actions:
            if action.act == Act.TL:
                yaw -= action.value
            elif action.act == Act.TR:
                yaw += action.value
            elif action.act in (Act.FWD, Act.BACK, Act.LEFT, Act.RIGHT):
                cam_delta = np.zeros(3, dtype=np.float32)
                if action.act == Act.FWD:
                    cam_delta[2] += action.value
                elif action.act == Act.BACK:
                    cam_delta[2] -= action.value
                elif action.act == Act.LEFT:
                    cam_delta[0] -= action.value
                elif action.act == Act.RIGHT:
                    cam_delta[0] += action.value
                R = rot_yaw_deg(yaw)
                t = t + R @ cam_delta
        R = rot_yaw_deg(yaw)
        c2ws.append(rt_to_c2w(R, t))
    return np.stack(c2ws, axis=0)


def yaw_from_rot(R: np.ndarray) -> float:
    return math.degrees(math.atan2(R[0, 2], R[2, 2]))


def actions_from_pose_delta(
    c2w_a: np.ndarray,
    c2w_b: np.ndarray,
    yaw_step_deg: float,
    trans_step_m: float,
    yaw_thresh_deg: float = 1.0,
    trans_thresh_m: float = 0.02,
) -> list[Action]:
    Ta = c2w_to_mat4(c2w_a)
    Tb = c2w_to_mat4(c2w_b)
    Ra = Ta[:3, :3]
    Rb = Tb[:3, :3]
    R_rel = Ra.T @ Rb
    yaw_deg = yaw_from_rot(R_rel)

    pa = Ta[:3, 3]
    pb = Tb[:3, 3]
    delta_world = pb - pa
    delta_cam = Ra.T @ delta_world

    actions: list[Action] = []
    if abs(yaw_deg) >= yaw_thresh_deg:
        actions.extend(quantize_actions(Act.TR, yaw_deg, yaw_step_deg))

    dx = float(delta_cam[0])
    dz = float(delta_cam[2])
    if max(abs(dx), abs(dz)) >= trans_thresh_m:
        if abs(dx) >= abs(dz):
            actions.extend(quantize_actions(Act.RIGHT, dx, trans_step_m))
        else:
            actions.extend(quantize_actions(Act.FWD, dz, trans_step_m))
    return actions
