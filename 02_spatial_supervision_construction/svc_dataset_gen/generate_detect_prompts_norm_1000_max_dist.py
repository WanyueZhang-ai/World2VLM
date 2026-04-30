from __future__ import annotations

import generate_detect_prompts_norm_1000 as base


def relevant_indices_max_dist(traj: base.TrajMeta) -> list[int]:
    if not traj.frame_paths:
        return []
    if len(traj.frame_paths) == 1:
        return [0]
    return [0, len(traj.frame_paths) - 1]


def pick_target_for_d3_max_dist(traj: base.TrajMeta) -> int | None:
    if len(traj.frame_paths) <= 1:
        return None
    return len(traj.frame_paths) - 1


base.relevant_indices = relevant_indices_max_dist
base._pick_target_for_d3 = pick_target_for_d3_max_dist


if __name__ == "__main__":
    base.main()
