from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from hyvideo import datagen_actions, datagen_io

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TRANSLATION_ACTIONS = {"forward", "backward", "shift_left", "shift_right"}
ROTATION_ACTIONS = {"turn_left", "turn_right"}


@dataclass(frozen=True)
class TrajectoryRequest:
    scene_id: str
    anchor_path: Path
    traj_id: str
    kind: str
    preset_name: str
    actions: tuple[datagen_actions.MetricAction, ...]
    scheduled: datagen_actions.ScheduledTrajectory
    prepared: datagen_actions.PreparedTrajectory
    pose_json: dict[str, Any]
    progress_points: tuple[dict[str, Any], ...]


class WorldPlayGenerationRuntime:
    def __init__(self, args: argparse.Namespace):
        from hyvideo import generate as generate_runtime

        runtime_args = argparse.Namespace(**vars(args))
        runtime_args.image_path = "__batch_i2v__"
        runtime_args.output_path = str(Path(args.output_root) / args.version / "_runtime_unused")
        runtime_args.pose = ""
        runtime_args.prompt = args.prompt
        self.args = runtime_args
        self.generate_runtime = generate_runtime
        self.pipe = generate_runtime.create_generation_pipeline(runtime_args)

    def generate(self, *, prompt: str, image_path: str, pose_data: dict[str, Any], video_length: int):
        return self.generate_runtime.run_generation_once(
            self.pipe,
            prompt=prompt,
            image_path=image_path,
            pose_data=pose_data,
            video_length=video_length,
            args=self.args,
        )

    def select_primary_video_tensor(self, out):
        return self.generate_runtime.select_primary_video_tensor(out, self.args.sr)

    def save_video(self, video_tensor, path: Path) -> None:
        self.generate_runtime.save_video(video_tensor, str(path))


def str_to_bool(value):
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def _round_to_grid_half_up(value: float, grid: float) -> float:
    steps = math.floor((abs(float(value)) / grid) + 0.5 + 1e-12)
    return round(steps * grid, 6)


def _jsonify(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonify(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonify(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_rng(args: argparse.Namespace) -> np.random.Generator:
    return np.random.default_rng(int(args.seed))


def scene_id_from_image(image_path: Path) -> str:
    return image_path.stem


def scan_input_images(input_dir: Path, max_images: int | None = None) -> list[Path]:
    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if max_images is not None:
        return image_paths[: max(0, int(max_images))]
    return image_paths


def _sample_quantized_value(
    rng: np.random.Generator,
    *,
    min_value: float,
    max_value: float,
    grid: float,
) -> float:
    if max_value < min_value:
        raise ValueError("max_value must be >= min_value")
    if grid <= 0:
        raise ValueError("grid must be positive")

    start = _round_to_grid_half_up(min_value, grid)
    end = _round_to_grid_half_up(max_value, grid)
    if end < start:
        end = start

    values: list[float] = []
    current = start
    while current <= end + 1e-9:
        values.append(round(current, 6))
        current += grid
    if not values:
        values = [round(start, 6)]
    return float(rng.choice(values))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}, got {type(payload)}")
    return payload


def _load_motion_config(args: argparse.Namespace) -> dict[str, Any]:
    cached = getattr(args, "_motion_config_payload", None)
    if cached is not None:
        return cached

    config_path = Path(args.motion_config).expanduser().resolve()
    payload = _load_yaml(config_path)
    setattr(args, "_motion_config_payload", payload)
    setattr(args, "_motion_config_path", config_path)
    return payload


def _sample_value_from_spec(
    action_name: str,
    spec: Mapping[str, Any],
    rng: np.random.Generator,
) -> float:
    if action_name in TRANSLATION_ACTIONS:
        range_key = "distance_range"
        default_unit = float(datagen_actions.TRANSLATION_GRID)
    elif action_name in ROTATION_ACTIONS:
        range_key = "angle_range"
        default_unit = float(datagen_actions.ROTATION_GRID_DEG)
    else:
        raise ValueError(f"Unsupported action: {action_name}")

    if range_key not in spec:
        raise ValueError(f"Missing '{range_key}' for action '{action_name}'")
    raw_range = spec[range_key]
    if not isinstance(raw_range, Sequence) or len(raw_range) != 2:
        raise ValueError(f"Expected 2-value sequence for '{range_key}' in action '{action_name}'")
    min_value, max_value = raw_range
    unit = float(spec.get("unit", default_unit))
    return _sample_quantized_value(
        rng,
        min_value=float(min_value),
        max_value=float(max_value),
        grid=unit,
    )


def _build_request(
    *,
    scene_id: str,
    anchor_path: Path,
    traj_id: str,
    kind: str,
    preset_name: str,
    actions: Sequence[datagen_actions.MetricAction],
) -> TrajectoryRequest:
    scheduled = datagen_actions.schedule_actions(actions)
    prepared = datagen_actions.prepare_trajectory_plan(scheduled)
    return TrajectoryRequest(
        scene_id=scene_id,
        anchor_path=anchor_path,
        traj_id=traj_id,
        kind=kind,
        preset_name=preset_name,
        actions=tuple(actions),
        scheduled=scheduled,
        prepared=prepared,
        pose_json=prepared.pose_json,
        progress_points=prepared.progress_points,
    )


def build_scene_requests(
    anchor_path: Path,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[TrajectoryRequest]:
    scene_id = scene_id_from_image(anchor_path)
    requests: list[TrajectoryRequest] = []
    motion_config = _load_motion_config(args)

    for single_spec in motion_config.get("single_step", []):
        action_name = str(single_spec["name"])
        sample_count = int(single_spec.get("count", 1))
        for sample_index in range(sample_count):
            value = _sample_value_from_spec(action_name, single_spec, rng)
            action = datagen_actions.sample_metric_action(action_name, value)
            requests.append(
                _build_request(
                    scene_id=scene_id,
                    anchor_path=anchor_path,
                    traj_id=f"{action_name}_{sample_index:02d}",
                    kind="single_step",
                    preset_name=action_name,
                    actions=(action,),
                )
            )

    for preset_index, preset_spec in enumerate(motion_config.get("multi_step_presets", [])):
        preset_name = str(preset_spec["name"])
        sample_count = int(preset_spec.get("count", 1))
        steps = preset_spec.get("steps", [])
        if not steps:
            raise ValueError(f"Multi-step preset '{preset_name}' must define non-empty steps")
        for sample_index in range(sample_count):
            sampled_actions = tuple(
                datagen_actions.sample_metric_action(
                    str(step_spec.get("act") or step_spec.get("name")),
                    _sample_value_from_spec(
                        str(step_spec.get("act") or step_spec.get("name")),
                        step_spec,
                        rng,
                    ),
                )
                for step_spec in steps
            )
            traj_id = f"multi_step_{preset_index:02d}_{preset_name}"
            if sample_count > 1:
                traj_id = f"{traj_id}_{sample_index:02d}"
            requests.append(
                _build_request(
                    scene_id=scene_id,
                    anchor_path=anchor_path,
                    traj_id=traj_id,
                    kind="multi_step",
                    preset_name=preset_name,
                    actions=sampled_actions,
                )
            )

    return requests


def _version_root(args: argparse.Namespace) -> Path:
    return Path(args.output_root).expanduser().resolve() / args.version


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _build_manifest_record(
    request: TrajectoryRequest,
    *,
    version_root: Path,
    sample_paths: Sequence[Path],
) -> dict[str, Any]:
    return {
        "scene_id": request.scene_id,
        "traj_id": request.traj_id,
        "kind": request.kind,
        "preset_name": request.preset_name,
        "anchor_path": str(request.anchor_path),
        "samples_rgb": [_relative_to_root(path, version_root) for path in sample_paths],
        "video_length": int(request.prepared.video_length),
        "actions": _jsonify(request.actions),
    }


def run_batch_generation(
    args: argparse.Namespace,
    runtime_factory: Callable[[argparse.Namespace], Any] | None = None,
) -> dict[str, int]:
    version_root = _version_root(args)
    datagen_io.ensure_dir(version_root)
    input_dir = Path(args.input_dir).expanduser().resolve()
    image_paths = scan_input_images(input_dir, args.max_images)
    if not image_paths:
        raise RuntimeError(f"No input images found under {input_dir}")

    manifest_path = version_root / "manifest.jsonl"
    if int(os.environ.get("RANK", "0")) == 0 and manifest_path.exists() and not args.skip_existing:
        manifest_path.unlink()

    runtime_factory = runtime_factory or WorldPlayGenerationRuntime
    runtime = runtime_factory(args)
    rng = build_rng(args)
    stats = {
        "scene_count": 0,
        "traj_requested": 0,
        "traj_generated": 0,
        "traj_skipped": 0,
        "traj_failed": 0,
        "sample_frame_count": 0,
    }

    for image_path in image_paths:
        scene_id = scene_id_from_image(image_path)
        scene_dir = datagen_io.prepare_scene_dir(version_root, scene_id)
        stats["scene_count"] += 1
        requests = build_scene_requests(image_path, args, rng)

        for request in requests:
            stats["traj_requested"] += 1
            traj_dir = scene_dir / f"traj_{request.traj_id}"
            traj_json_path = traj_dir / "traj.json"
            if args.skip_existing and traj_json_path.exists():
                stats["traj_skipped"] += 1
                continue

            try:
                out = runtime.generate(
                    prompt=args.prompt,
                    image_path=str(image_path),
                    pose_data=request.pose_json,
                    video_length=request.prepared.video_length,
                )
                video_tensor = runtime.select_primary_video_tensor(out)
                sample_indices = datagen_io.select_sample_indices(request.progress_points)

                if int(os.environ.get("RANK", "0")) == 0:
                    datagen_io.prepare_traj_dir(scene_dir, request.traj_id)
                    pose_path = datagen_io.write_json(traj_dir / "pose.json", request.pose_json)
                    sample_paths = datagen_io.save_selected_frames(video_tensor, sample_indices, traj_dir)
                    mp4_path = None
                    if args.save_mp4:
                        mp4_path = traj_dir / "gen.mp4"
                        runtime.save_video(video_tensor, mp4_path)
                    datagen_io.write_traj_json(
                        traj_dir,
                        prompt=args.prompt,
                        anchor_path=request.anchor_path,
                        actions=request.actions,
                        latent_chunks=request.scheduled.chunks,
                        video_length=request.prepared.video_length,
                        retained_frame_indices=sample_indices,
                        progress_points=request.progress_points,
                        sample_paths=sample_paths,
                        pose_path=pose_path,
                        mp4_path=mp4_path,
                    )
                    datagen_io.append_manifest_record(
                        manifest_path,
                        _build_manifest_record(
                            request,
                            version_root=version_root,
                            sample_paths=sample_paths,
                        ),
                    )
                    stats["sample_frame_count"] += len(sample_paths)
                stats["traj_generated"] += 1
            except Exception:
                stats["traj_failed"] += 1
                continue

    if int(os.environ.get("RANK", "0")) == 0:
        datagen_io.write_stats(version_root / "stats.json", stats)

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-generate HY-WorldPlay viewpoint data")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--save_mp4", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--motion_config", type=str, required=True)

    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--resolution", type=str, required=True, choices=["480p", "720p"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--action_ckpt", type=str, required=True)
    parser.add_argument("--aspect_ratio", type=str, default="16:9")
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--sr", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument(
        "--save_pre_sr_video",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--rewrite", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--offloading", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument(
        "--group_offloading",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=None,
    )
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--few_step", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--model_type", type=str, required=True, choices=["bi", "ar"])
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument(
        "--enable_torch_compile",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--use_sageattn", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--sage_blocks_range", type=str, default="0-53")
    parser.add_argument(
        "--use_vae_parallel",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--use_fp8_gemm", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--quant_type", type=str, default="fp8-per-block")
    parser.add_argument("--include_patterns", type=str, default="double_blocks")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_batch_generation(args)


if __name__ == "__main__":
    main()
