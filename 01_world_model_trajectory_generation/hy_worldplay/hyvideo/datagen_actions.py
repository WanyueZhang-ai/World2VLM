from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from hyvideo.generate_custom_trajectory import generate_camera_trajectory_local

DEFAULT_INTRINSIC = [
    [969.6969696969696, 0.0, 960.0],
    [0.0, 969.6969696969696, 540.0],
    [0.0, 0.0, 1.0],
]

TRANSLATION_GRID = 0.1
ROTATION_GRID_DEG = 10.0

SUPPORTED_ACTIONS = (
    "forward",
    "backward",
    "turn_left",
    "turn_right",
    "shift_left",
    "shift_right",
)

_ACTION_METADATA: Mapping[str, Mapping[str, Any]] = {
    "forward": {
        "kind": "translation",
        "grid": TRANSLATION_GRID,
        "sign": 1.0,
        "motion_key": "forward",
    },
    "backward": {
        "kind": "translation",
        "grid": TRANSLATION_GRID,
        "sign": -1.0,
        "motion_key": "forward",
    },
    "shift_left": {
        "kind": "translation",
        "grid": TRANSLATION_GRID,
        "sign": -1.0,
        "motion_key": "right",
    },
    "shift_right": {
        "kind": "translation",
        "grid": TRANSLATION_GRID,
        "sign": 1.0,
        "motion_key": "right",
    },
    "turn_left": {
        "kind": "rotation",
        "grid": ROTATION_GRID_DEG,
        "sign": 1.0,
        "motion_key": "yaw",
    },
    "turn_right": {
        "kind": "rotation",
        "grid": ROTATION_GRID_DEG,
        "sign": -1.0,
        "motion_key": "yaw",
    },
}


@dataclass(frozen=True)
class MetricAction:
    name: str
    value: float

    def __post_init__(self) -> None:
        validate_action_name(self.name)
        if not np.isfinite(self.value):
            raise ValueError("action value must be finite")
        if self.value < 0:
            raise ValueError("action value must be non-negative")

    @property
    def kind(self) -> str:
        return str(_ACTION_METADATA[self.name]["kind"])


@dataclass(frozen=True)
class LatentChunk:
    action: MetricAction
    latent_values: tuple[float, ...]


@dataclass(frozen=True)
class ScheduledTrajectory:
    actions: tuple[MetricAction, ...]
    chunks: tuple[LatentChunk, ...]
    total_latents: int
    video_length: int


@dataclass(frozen=True)
class PreparedTrajectory:
    scheduled: ScheduledTrajectory
    pose_json: dict[str, dict[str, list[list[float]] | list[list[float]]]]
    progress_points: tuple[dict[str, float | int | bool | str], ...]
    padded_latents: int
    video_length: int


def validate_action_name(name: str) -> str:
    if name not in _ACTION_METADATA:
        raise ValueError(
            f"Unsupported action '{name}'. Supported actions: {', '.join(SUPPORTED_ACTIONS)}"
        )
    return name


def _round_to_grid(value: float, grid: float) -> float:
    if grid <= 0:
        raise ValueError("grid must be positive")
    if not np.isfinite(value):
        raise ValueError("value must be finite")
    # Use explicit half-up quantization so 0.15m -> 0.2m and 25deg -> 30deg
    # instead of relying on banker's rounding or float-representation artifacts.
    quantized_steps = math.floor((abs(float(value)) / grid) + 0.5 + 1e-12)
    quantized = quantized_steps * grid
    if abs(quantized) < 1e-9:
        return 0.0
    return round(abs(quantized), 6)


def sample_metric_action(name: str, value: float) -> MetricAction:
    validated = validate_action_name(name)
    grid = float(_ACTION_METADATA[validated]["grid"])
    return MetricAction(name=validated, value=_round_to_grid(value, grid))


def sample_action_sequence(raw_actions: Sequence[tuple[str, float]]) -> tuple[MetricAction, ...]:
    return tuple(sample_metric_action(name, value) for name, value in raw_actions)


def sample_single_action(name: str, value: float) -> MetricAction:
    return sample_metric_action(name, value)


def sample_multi_step_actions(steps: Sequence[Mapping[str, float]]) -> tuple[MetricAction, ...]:
    actions: list[MetricAction] = []
    for step in steps:
        name = str(step["name"])
        value = float(step["value"])
        actions.append(sample_metric_action(name, value))
    return tuple(actions)


def build_latent_values(value: float, base_step: float, direction: float = 1.0) -> tuple[float, ...]:
    if base_step <= 0:
        raise ValueError("base_step must be positive")
    magnitude = abs(float(value))
    if magnitude < 1e-9:
        return ()

    full_steps = int(np.floor((magnitude + 1e-9) / base_step))
    remainder = magnitude - full_steps * base_step
    latents = [base_step] * full_steps
    if remainder > 1e-6 or not latents:
        latents.append(remainder if remainder > 1e-6 else base_step)

    sign = 1.0 if direction >= 0 else -1.0
    return tuple(round(sign * step, 6) for step in latents)


def compute_video_length(total_latents: int) -> int:
    if total_latents < 0:
        raise ValueError("total_latents must be non-negative")
    return 1 + 4 * total_latents


def schedule_actions(
    actions: Sequence[MetricAction],
    translation_step: float = 0.08,
    rotation_step_deg: float = 3.0,
) -> ScheduledTrajectory:
    chunks: list[LatentChunk] = []
    total_latents = 0
    normalized_actions = tuple(actions)

    for action in normalized_actions:
        metadata = _ACTION_METADATA[action.name]
        base_step = (
            float(translation_step)
            if metadata["kind"] == "translation"
            else float(rotation_step_deg)
        )
        values = build_latent_values(action.value, base_step, direction=float(metadata["sign"]))
        chunks.append(LatentChunk(action=action, latent_values=values))
        total_latents += len(values)

    return ScheduledTrajectory(
        actions=normalized_actions,
        chunks=tuple(chunks),
        total_latents=total_latents,
        video_length=compute_video_length(total_latents),
    )


def scheduled_actions_to_motions(scheduled: ScheduledTrajectory) -> list[dict[str, float]]:
    motions: list[dict[str, float]] = []
    for chunk in scheduled.chunks:
        metadata = _ACTION_METADATA[chunk.action.name]
        motion_key = str(metadata["motion_key"])
        for latent_value in chunk.latent_values:
            if motion_key == "yaw":
                motions.append({"yaw": float(np.deg2rad(latent_value))})
            else:
                motions.append({motion_key: float(latent_value)})
    return motions


def scheduled_actions_to_pose_json(
    scheduled: ScheduledTrajectory,
) -> dict[str, dict[str, list[list[float]] | list[list[float]]]]:
    motions = scheduled_actions_to_motions(scheduled)
    poses = generate_camera_trajectory_local(motions)
    pose_json: dict[str, dict[str, list[list[float]] | list[list[float]]]] = {}
    for index, pose in enumerate(poses):
        pose_json[str(index)] = {
            "extrinsic": pose.tolist(),
            "K": DEFAULT_INTRINSIC,
        }
    return pose_json


def scheduled_actions_to_progress_points(
    scheduled: ScheduledTrajectory,
) -> list[dict[str, float | int | bool | str]]:
    progress_points: list[dict[str, float | int | bool | str]] = [
        {
            "frame_index": 0,
            "action_index": -1,
            "action_name": "start",
            "metric_kind": "translation",
            "metric_value": 0.0,
            "distance_m": 0.0,
            "angle_deg": 0.0,
            "is_action_end": False,
        }
    ]
    total_distance = 0.0
    total_angle = 0.0
    latent_index = 0

    for action_index, chunk in enumerate(scheduled.chunks):
        segment_value = 0.0
        for value_index, latent_value in enumerate(chunk.latent_values):
            magnitude = abs(float(latent_value))
            segment_value += magnitude
            if chunk.action.kind == "translation":
                total_distance += magnitude
            else:
                total_angle += magnitude
            latent_index += 1
            progress_points.append(
                {
                    "frame_index": latent_index * 4,
                    "action_index": action_index,
                    "action_name": chunk.action.name,
                    "metric_kind": chunk.action.kind,
                    "metric_value": round(segment_value, 6),
                    "distance_m": round(total_distance, 6),
                    "angle_deg": round(total_angle, 6),
                    "is_action_end": value_index == len(chunk.latent_values) - 1,
                }
            )

    return progress_points


def prepare_trajectory_plan(
    scheduled: ScheduledTrajectory,
    latent_multiple: int = 4,
) -> PreparedTrajectory:
    if latent_multiple <= 0:
        raise ValueError("latent_multiple must be positive")

    pose_json = scheduled_actions_to_pose_json(scheduled)
    progress_points = scheduled_actions_to_progress_points(scheduled)

    pose_count = len(pose_json)
    padded_pose_count = (
        (pose_count + latent_multiple - 1) // latent_multiple
    ) * latent_multiple
    padded_latents = padded_pose_count - pose_count

    if padded_latents:
        last_pose = copy.deepcopy(pose_json[str(pose_count - 1)])
        last_progress = progress_points[-1]
        for extra_index in range(padded_latents):
            pose_json[str(pose_count + extra_index)] = copy.deepcopy(last_pose)
            progress_points.append(
                {
                    "frame_index": int(last_progress["frame_index"]) + 4 * (extra_index + 1),
                    "action_index": int(last_progress["action_index"]),
                    "action_name": "hold",
                    "metric_kind": "hold",
                    "metric_value": float(last_progress["metric_value"]),
                    "distance_m": float(last_progress["distance_m"]),
                    "angle_deg": float(last_progress["angle_deg"]),
                    "is_action_end": False,
                }
            )

    return PreparedTrajectory(
        scheduled=scheduled,
        pose_json=pose_json,
        progress_points=tuple(progress_points),
        padded_latents=padded_latents,
        video_length=len(pose_json) * 4 - 3,
    )
