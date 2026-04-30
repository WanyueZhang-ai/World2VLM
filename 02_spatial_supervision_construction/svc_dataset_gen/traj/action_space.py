from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Act(Enum):
    TL = "TURN_LEFT"
    TR = "TURN_RIGHT"
    FWD = "GO_FORWARD"
    BACK = "GO_BACK"
    LEFT = "GO_LEFT"
    RIGHT = "GO_RIGHT"


@dataclass(frozen=True)
class Action:
    act: Act
    value: float


_SHORT_NAME = {
    Act.TL: "TL",
    Act.TR: "TR",
    Act.FWD: "FWD",
    Act.BACK: "BACK",
    Act.LEFT: "LEFT",
    Act.RIGHT: "RIGHT",
}


def format_value(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    text = f"{value:.3f}"
    return text.rstrip("0").rstrip(".")


def action_to_str(action: Action) -> str:
    return f"{_SHORT_NAME[action.act]}_{format_value(action.value)}"


def actions_to_strings(actions: list[Action]) -> list[str]:
    return [action_to_str(a) for a in actions]


def _axis_and_signed_value(action: Action) -> tuple[str, float]:
    if action.act == Act.TL:
        return ("yaw", action.value)
    if action.act == Act.TR:
        return ("yaw", -action.value)
    if action.act == Act.LEFT:
        return ("x", action.value)
    if action.act == Act.RIGHT:
        return ("x", -action.value)
    if action.act == Act.FWD:
        return ("z", action.value)
    if action.act == Act.BACK:
        return ("z", -action.value)
    raise ValueError(f"Unsupported action {action.act}")


def _action_from_axis(axis: str, signed_value: float) -> Action | None:
    if axis == "yaw":
        if signed_value > 0:
            return Action(Act.TL, abs(signed_value))
        return Action(Act.TR, abs(signed_value))
    if axis == "x":
        if signed_value > 0:
            return Action(Act.LEFT, abs(signed_value))
        return Action(Act.RIGHT, abs(signed_value))
    if axis == "z":
        if signed_value > 0:
            return Action(Act.FWD, abs(signed_value))
        return Action(Act.BACK, abs(signed_value))
    return None


def canonicalize(
    actions: list[Action],
    yaw_eps_deg: float = 1.0,
    trans_eps_m: float = 0.02,
) -> list[Action]:
    """
    Canonicalize a list of actions by merging adjacent actions with the same axis.
    """
    stack: list[Action] = []
    for action in actions:
        axis, signed_value = _axis_and_signed_value(action)
        if abs(signed_value) < (yaw_eps_deg if axis == "yaw" else trans_eps_m):
            continue
        if stack:
            last_axis, last_signed = _axis_and_signed_value(stack[-1])
            if last_axis == axis:
                signed_value += last_signed
                stack.pop()
        if abs(signed_value) < (yaw_eps_deg if axis == "yaw" else trans_eps_m):
            continue
        merged = _action_from_axis(axis, signed_value)
        if merged is not None:
            stack.append(merged)
    return stack


def quantize_actions(
    act: Act,
    total_value: float,
    step: float,
) -> list[Action]:
    if step <= 0:
        raise ValueError("step must be positive")
    n = int(round(abs(total_value) / step))
    if n <= 0:
        return []
    base_act = act
    if total_value < 0:
        if act == Act.TL:
            base_act = Act.TR
        elif act == Act.TR:
            base_act = Act.TL
        elif act == Act.FWD:
            base_act = Act.BACK
        elif act == Act.BACK:
            base_act = Act.FWD
        elif act == Act.LEFT:
            base_act = Act.RIGHT
        elif act == Act.RIGHT:
            base_act = Act.LEFT
    return [Action(base_act, step) for _ in range(n)]
