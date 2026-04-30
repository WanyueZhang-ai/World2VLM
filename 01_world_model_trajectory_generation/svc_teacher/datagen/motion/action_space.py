from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Act(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    SHIFT_LEFT = "shift_left"
    SHIFT_RIGHT = "shift_right"


@dataclass(frozen=True)
class ActionSpec:
    act: Act
    magnitude: float

    def __post_init__(self) -> None:
        if not isinstance(self.act, Act):
            raise TypeError("act must be an Act enum value")
        if not math.isfinite(self.magnitude):
            raise ValueError("magnitude must be finite")
        if self.magnitude < 0:
            raise ValueError("magnitude must be non-negative")


_SHORT_NAME = {
    Act.TURN_LEFT: "TL",
    Act.TURN_RIGHT: "TR",
    Act.FORWARD: "FWD",
    Act.BACKWARD: "BACK",
    Act.SHIFT_LEFT: "LEFT",
    Act.SHIFT_RIGHT: "RIGHT",
}


def is_rotation_action(act: Act) -> bool:
    return act in {Act.TURN_LEFT, Act.TURN_RIGHT}


def format_magnitude(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    text = f"{value:.6f}"
    return text.rstrip("0").rstrip(".")


def action_to_token(action: ActionSpec) -> str:
    return f"{_SHORT_NAME[action.act]}_{format_magnitude(action.magnitude)}"


def actions_to_tokens(actions: list[ActionSpec]) -> list[str]:
    return [action_to_token(action) for action in actions]


def sample_magnitude(
    rng: np.random.Generator,
    min_val: float,
    max_val: float,
    unit: float,
) -> float:
    """Uniform random sample in [min_val, max_val] with discrete unit."""
    if unit <= 0:
        raise ValueError("unit must be positive")
    if max_val < min_val:
        raise ValueError("max_val must be >= min_val")
    steps = int(round((max_val - min_val) / unit))
    choice = int(rng.integers(0, steps + 1))
    return round(min_val + choice * unit, 6)


def _axis_and_signed_value(action: ActionSpec) -> tuple[str, float]:
    if action.act == Act.TURN_LEFT:
        return ("yaw", action.magnitude)
    if action.act == Act.TURN_RIGHT:
        return ("yaw", -action.magnitude)
    if action.act == Act.SHIFT_LEFT:
        return ("x", action.magnitude)
    if action.act == Act.SHIFT_RIGHT:
        return ("x", -action.magnitude)
    if action.act == Act.FORWARD:
        return ("z", action.magnitude)
    if action.act == Act.BACKWARD:
        return ("z", -action.magnitude)
    raise ValueError(f"Unsupported action {action.act}")


def _action_from_axis(axis: str, signed_value: float) -> ActionSpec | None:
    if axis == "yaw":
        if signed_value > 0:
            return ActionSpec(Act.TURN_LEFT, abs(signed_value))
        return ActionSpec(Act.TURN_RIGHT, abs(signed_value))
    if axis == "x":
        if signed_value > 0:
            return ActionSpec(Act.SHIFT_LEFT, abs(signed_value))
        return ActionSpec(Act.SHIFT_RIGHT, abs(signed_value))
    if axis == "z":
        if signed_value > 0:
            return ActionSpec(Act.FORWARD, abs(signed_value))
        return ActionSpec(Act.BACKWARD, abs(signed_value))
    return None


def canonicalize(
    actions: list[ActionSpec],
    yaw_eps_deg: float = 1.0,
    trans_eps_m: float = 0.02,
) -> list[ActionSpec]:
    """Merge adjacent actions with the same axis."""
    stack: list[ActionSpec] = []
    for action in actions:
        axis, signed_value = _axis_and_signed_value(action)
        threshold = yaw_eps_deg if axis == "yaw" else trans_eps_m
        if abs(signed_value) < threshold:
            continue
        if stack:
            last_axis, last_signed = _axis_and_signed_value(stack[-1])
            if last_axis == axis:
                signed_value += last_signed
                stack.pop()
        if abs(signed_value) < threshold:
            continue
        merged = _action_from_axis(axis, signed_value)
        if merged is not None:
            stack.append(merged)
    return stack

