from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate undetect prompt data (A1-A4)")
    parser.add_argument("--data_root", type=str, default="outputs/scenes")
    parser.add_argument("--templates", type=str, default="datagen/prompt/templates.jsonl")
    parser.add_argument("--out_path", type=str, default="outputs/data_undetect.jsonl")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--k_per_traj", type=int, default=3)
    return parser


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    text = f"{value:.6f}"
    return text.rstrip("0").rstrip(".")


def _parse_action(action: str) -> tuple[str, float]:
    try:
        name, raw_value = action.split("_", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid action string: {action}") from exc
    value = float(raw_value)
    if name == "TL":
        return ("yaw", value)
    if name == "TR":
        return ("yaw", -value)
    if name == "FWD":
        return ("z", value)
    if name == "BACK":
        return ("z", -value)
    if name == "LEFT":
        return ("x", value)
    if name == "RIGHT":
        return ("x", -value)
    raise ValueError(f"Unknown action type: {name}")


def _sum_actions(actions: list[str]) -> dict[str, float]:
    sums = {"yaw": 0.0, "x": 0.0, "z": 0.0}
    for action in actions:
        axis, value = _parse_action(action)
        sums[axis] += value
    return sums


def _move_action(axis: str, signed_value: float) -> dict[str, Any] | None:
    if abs(signed_value) < 1e-6:
        return None
    if axis == "z":
        direction = "forward" if signed_value > 0 else "backward"
    elif axis == "x":
        direction = "left" if signed_value > 0 else "right"
    else:
        return None
    dist = abs(signed_value)
    return {
        "kind": "move",
        "axis": axis,
        "signed": signed_value,
        "dir": direction,
        "value": dist,
        "text": f"move {direction} {_format_number(dist)} meters",
    }


def _turn_action(signed_value: float) -> dict[str, Any] | None:
    if abs(signed_value) < 1e-6:
        return None
    direction = "left" if signed_value > 0 else "right"
    angle = abs(signed_value)
    return {
        "kind": "turn",
        "axis": "yaw",
        "signed": signed_value,
        "dir": direction,
        "value": angle,
        "text": f"turn {direction} {_format_number(angle)} degrees",
    }


def _action_from_token(token: str) -> dict[str, Any] | None:
    axis, value = _parse_action(token)
    if axis == "yaw":
        return _turn_action(value)
    return _move_action(axis, value)


def _actions_from_prefix(prefix: list[str]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for token in prefix:
        action = _action_from_token(token)
        if action:
            actions.append(action)
    return actions


def _format_action_seq(actions: list[dict[str, Any]], sep: str = "; ") -> str:
    return sep.join(action["text"] for action in actions)


def _format_action_claim(actions: list[dict[str, Any]]) -> str:
    texts = [action["text"] for action in actions]
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]
    if len(texts) == 2:
        return f"{texts[0]} and {texts[1]}"
    return ", ".join(texts[:-1]) + f" and {texts[-1]}"


def _parse_templates(path: Path) -> dict[str, dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    lines = [line for line in raw.splitlines() if not line.strip().startswith("//")]
    content = "\n".join(lines)

    objects: list[dict[str, Any]] = []
    start = None
    depth = 0
    in_string = False
    escape = False

    for idx, ch in enumerate(content):
        if in_string:
            if escape:
                escape = False
            else:
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                chunk = content[start : idx + 1].strip()
                if chunk:
                    objects.append(json.loads(chunk))
                start = None

    if depth != 0 or len(objects) < 4:
        raise ValueError("Failed to parse A1-A4 templates from templates.jsonl")
    return {"A1": objects[0], "A2": objects[1], "A3": objects[2], "A4": objects[3]}


def _fill_template(template: dict[str, Any], replacements: dict[str, str]) -> dict[str, Any]:
    data = json.loads(json.dumps(template))
    for msg in data.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        for key, value in replacements.items():
            content = content.replace("{" + key + "}", value)
        msg["content"] = content
    images = []
    for img in data.get("images", []):
        for key, value in replacements.items():
            img = img.replace("{" + key + "}", value)
        images.append(img)
    return {
        "task_type": data.get("task_type", []),
        "messages": data.get("messages", []),
        "images": images,
    }


def _even_indices(num_frames: int, k: int) -> list[int]:
    if num_frames <= 1 or k <= 0:
        return []
    max_idx = num_frames - 1
    target = min(k, max_idx)
    base = [int(round((i + 1) * max_idx / (target + 1))) for i in range(target)]
    base = [min(max_idx, max(1, idx)) for idx in base]
    out: list[int] = []
    used: set[int] = set()
    for idx in base:
        if idx not in used:
            out.append(idx)
            used.add(idx)
            continue
        for delta in range(1, max_idx + 1):
            for cand in (idx + delta, idx - delta):
                if 1 <= cand <= max_idx and cand not in used:
                    out.append(cand)
                    used.add(cand)
                    break
            else:
                continue
            break
    return out


def _choose_move(sums: dict[str, float]) -> dict[str, Any] | None:
    if abs(sums["z"]) >= abs(sums["x"]):
        return _move_action("z", sums["z"])
    return _move_action("x", sums["x"])


def _perturb_signed(signed: float, step: float, rng: random.Random) -> float:
    if abs(signed) < 1e-6:
        return step
    if rng.random() < 0.5:
        return -signed
    return signed + (step if signed > 0 else -step)


def _normalize_image_path(path: Path) -> str:
    return str(path.resolve())


def _step_size_for_action(action: dict[str, Any], traj_params: dict[str, Any]) -> float:
    if action["kind"] == "turn":
        return float(
            traj_params.get(
                "max_yaw_step_per_frame",
                traj_params.get("max_step_per_frame", 5.0),
            )
        )
    return float(
        traj_params.get(
            "max_trans_step_per_frame",
            traj_params.get("max_step_per_frame", 0.2),
        )
    )


def _make_false_action(
    action: dict[str, Any],
    traj_params: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any] | None:
    signed = float(action["signed"])
    axis = action["axis"]
    step = _step_size_for_action(action, traj_params)
    signed_false = _perturb_signed(signed, step, rng)
    if action["kind"] == "turn":
        return _turn_action(signed_false)
    return _move_action(axis, signed_false)


def _make_false_actions(
    actions: list[dict[str, Any]],
    traj_params: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not actions:
        return []
    pick = rng.randrange(len(actions))
    false_action = _make_false_action(actions[pick], traj_params, rng)
    if not false_action:
        return []
    out = list(actions)
    out[pick] = false_action
    return out


def _build_single_step_prompts(
    frame_paths: list[str],
    actions_prefix: list[list[str]],
    templates: dict[str, dict[str, Any]],
    traj_params: dict[str, Any],
    k_per_traj: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompts: list[dict[str, Any]] = []
    a4_candidates: list[dict[str, Any]] = []
    kind = str(traj_params.get("action", ""))
    is_turn = kind in {"turn_left", "turn_right"}

    used: set[int] = set()
    for idx in _even_indices(len(frame_paths), k_per_traj):
        if not frame_paths[0] or not frame_paths[idx]:
            continue
        sums = _sum_actions(actions_prefix[idx])
        if is_turn:
            action = _turn_action(sums["yaw"])
            if not action:
                continue
            prompts.append(
                _fill_template(
                    templates["A2"],
                    {
                        "IMG_1": frame_paths[0],
                        "IMG_2": frame_paths[idx],
                        "TURN_DIR": action["dir"],
                        "ANGLE": _format_number(action["value"]),
                    },
                )
            )
        else:
            action = _choose_move(sums)
            if not action:
                continue
            prompts.append(
                _fill_template(
                    templates["A1"],
                    {
                        "IMG_1": frame_paths[0],
                        "IMG_2": frame_paths[idx],
                        "DIR": action["dir"],
                        "DIST": _format_number(action["value"]),
                    },
                )
            )
        used.add(idx)

    for idx in range(1, len(frame_paths)):
        if idx in used:
            continue
        if not frame_paths[0] or not frame_paths[idx]:
            continue
        sums = _sum_actions(actions_prefix[idx])
        action = _turn_action(sums["yaw"]) if is_turn else _choose_move(sums)
        if not action:
            continue
        a4_candidates.append(
            {
                "img1": frame_paths[0],
                "img2": frame_paths[idx],
                "actions": [action],
                "traj_params": traj_params,
            }
        )
    return prompts, a4_candidates


def _build_multi_step_prompts(
    frame_paths: list[str],
    actions_prefix: list[list[str]],
    templates: dict[str, dict[str, Any]],
    traj_params: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompts: list[dict[str, Any]] = []
    a4_candidates: list[dict[str, Any]] = []
    key_indices = sorted(set(int(i) for i in traj_params.get("key_frame_indices", [])))
    valid = [i for i in key_indices if 0 < i < len(frame_paths)]
    if not valid:
        valid = [i for i in range(1, len(frame_paths))]

    a3_idx = None
    a3_actions: list[dict[str, Any]] | None = None
    for idx in valid:
        if not frame_paths[0] or not frame_paths[idx]:
            continue
        actions = _actions_from_prefix(actions_prefix[idx])
        if 2 <= len(actions) <= 3:
            a3_idx = idx
            a3_actions = actions
            break
    if a3_idx is not None and a3_actions:
        prompts.append(
            _fill_template(
                templates["A3"],
                {
                    "IMG_1": frame_paths[0],
                    "IMG_2": frame_paths[a3_idx],
                    "ACTION_SEQ": _format_action_seq(a3_actions),
                },
            )
        )

    for idx in valid:
        if idx == a3_idx:
            continue
        if not frame_paths[0] or not frame_paths[idx]:
            continue
        actions = _actions_from_prefix(actions_prefix[idx])
        if not actions:
            continue
        a4_candidates.append(
            {
                "img1": frame_paths[0],
                "img2": frame_paths[idx],
                "actions": actions,
                "traj_params": traj_params,
            }
        )
    return prompts, a4_candidates


def main() -> None:
    args = build_arg_parser().parse_args()
    rng = random.Random(args.seed)

    data_root = Path(args.data_root).expanduser().resolve()
    templates = _parse_templates(Path(args.templates).expanduser().resolve())
    out_path = Path(args.out_path).expanduser().resolve()

    traj_paths = sorted(data_root.glob("*/traj_*/traj.json"))
    all_prompts: list[dict[str, Any]] = []
    a4_candidates: list[dict[str, Any]] = []

    for traj_json in traj_paths:
        payload = json.loads(traj_json.read_text(encoding="utf-8"))
        actions_prefix = payload.get("actions_prefix", [])
        frames = payload.get("transforms", {}).get("frames", [])
        if not actions_prefix or not frames:
            continue

        count = min(len(actions_prefix), len(frames))
        if count < 2:
            continue
        actions_prefix = actions_prefix[:count]
        frames = frames[:count]

        traj_dir = traj_json.parent
        frame_paths: list[str] = []
        for frame in frames:
            file_path = frame.get("file_path")
            if not file_path:
                frame_paths.append("")
                continue
            resolved = traj_dir / file_path.lstrip("./")
            frame_paths.append(_normalize_image_path(resolved))
        if not frame_paths[0]:
            continue

        traj_params = payload.get("traj_params", {})
        kind = str(traj_params.get("kind", ""))
        if kind == "single_step":
            prompts, candidates = _build_single_step_prompts(
                frame_paths=frame_paths,
                actions_prefix=actions_prefix,
                templates=templates,
                traj_params=traj_params,
                k_per_traj=args.k_per_traj,
            )
        else:
            prompts, candidates = _build_multi_step_prompts(
                frame_paths=frame_paths,
                actions_prefix=actions_prefix,
                templates=templates,
                traj_params=traj_params,
            )
        all_prompts.extend(prompts)
        a4_candidates.extend(candidates)

    rng.shuffle(a4_candidates)
    for idx, candidate in enumerate(a4_candidates):
        is_true = idx % 2 == 0
        actions = candidate["actions"]
        if is_true:
            claim_action = _format_action_claim(actions)
            tf = "true"
        else:
            false_actions = _make_false_actions(actions, candidate["traj_params"], rng)
            if not false_actions:
                continue
            claim_action = _format_action_claim(false_actions)
            tf = "false"
        if not claim_action:
            continue
        all_prompts.append(
            _fill_template(
                templates["A4"],
                {
                    "IMG_1": candidate["img1"],
                    "IMG_2": candidate["img2"],
                    "CLAIM_ACTION": claim_action,
                    "TF": tf,
                },
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in all_prompts:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")
    print(f"Wrote {len(all_prompts)} records to {out_path}")


if __name__ == "__main__":
    main()

