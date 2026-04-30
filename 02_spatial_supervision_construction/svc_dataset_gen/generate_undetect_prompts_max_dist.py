from __future__ import annotations

import random
from typing import Any

import generate_undetect_prompts as base


def _last_valid_index(frame_paths: list[str]) -> int | None:
    for idx in range(len(frame_paths) - 1, 0, -1):
        if frame_paths[idx]:
            return idx
    return None


def build_a1_a3_prompts_max_dist(
    traj_type: str,
    frame_paths: list[str],
    actions_prefix: list[list[str]],
    templates: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    k_per_traj: int,
) -> tuple[list[dict[str, Any]], set[int]]:
    del k_per_traj
    prompts: list[dict[str, Any]] = []
    used: set[int] = set()
    idx = _last_valid_index(frame_paths)
    if idx is None or not frame_paths[0]:
        return prompts, used

    sums = base._sum_actions(actions_prefix[idx])
    if traj_type == "T1":
        turn = base._turn_action(sums["yaw"])
        if not turn:
            return prompts, used
        prompts.append(
            base._fill_template(
                base._pick_template(templates, "A2", rng),
                {
                    "IMG_1": frame_paths[0],
                    "IMG_2": frame_paths[idx],
                    "TURN_DIR": turn["dir"],
                    "ANGLE": base._format_number(turn["value"]),
                    "TURN_CHOICE": base._turn_choice(turn["dir"]),
                },
            )
        )
        used.add(idx)
        return prompts, used

    if traj_type in {"T2", "T3"}:
        move = base._choose_move(sums)
        if not move:
            return prompts, used
        prompts.append(
            base._fill_template(
                base._pick_template(templates, "A1", rng),
                {
                    "IMG_1": frame_paths[0],
                    "IMG_2": frame_paths[idx],
                    "DIR": move["dir"],
                    "DIST": base._format_number(move["value"]),
                    "DIR_CHOICE": base._dir_choice(move["dir"]),
                },
            )
        )
        used.add(idx)
    return prompts, used


def build_a4_candidates_max_dist(
    traj_type: str,
    frame_paths: list[str],
    actions_prefix: list[list[str]],
    used_indices: set[int],
    traj_params: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    del rng
    idx = _last_valid_index(frame_paths)
    if idx is None or idx in used_indices or not frame_paths[0]:
        return []

    sums = base._sum_actions(actions_prefix[idx])
    if traj_type == "T1":
        action = base._turn_action(sums["yaw"])
    elif traj_type in {"T2", "T3"}:
        action = base._choose_move(sums)
    else:
        action = None
    if not action:
        return []
    return [
        {
            "img1": frame_paths[0],
            "img2": frame_paths[idx],
            "actions": [action],
            "traj_params": traj_params,
        }
    ]


def build_t4_prompts_max_dist(
    frame_paths: list[str],
    actions_prefix: list[list[str]],
    templates: dict[str, list[dict[str, Any]]],
    traj_params: dict[str, Any],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompts: list[dict[str, Any]] = []
    a4_candidates: list[dict[str, Any]] = []
    idx = _last_valid_index(frame_paths)
    if idx is None or not frame_paths[0]:
        return prompts, a4_candidates

    actions = base._actions_from_prefix(actions_prefix[idx])
    if not actions:
        return prompts, a4_candidates

    prompts.append(
        base._fill_template(
            base._pick_template(templates, "A3", rng),
            {
                "IMG_1": frame_paths[0],
                "IMG_2": frame_paths[idx],
                "ACTION_SEQ": base._format_action_seq(actions),
                "ACTION_COUNT": str(len(actions)),
                "ACTION_SEQ_NL": base._action_seq_to_natural(actions),
            },
        )
    )
    a4_candidates.append(
        {
            "img1": frame_paths[0],
            "img2": frame_paths[idx],
            "actions": actions,
            "traj_params": traj_params,
        }
    )
    return prompts, a4_candidates


base._build_a1_a3_prompts = build_a1_a3_prompts_max_dist
base._build_a4_candidates = build_a4_candidates_max_dist
base._build_t4_prompts = build_t4_prompts_max_dist


if __name__ == "__main__":
    base.main()
