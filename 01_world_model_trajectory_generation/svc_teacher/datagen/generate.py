from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from datagen.adapters import SVCAdapter, WorldFMAdapter
from datagen.io.schema import frame_record
from datagen.io.writer import (
    append_jsonl,
    merge_actions_between,
    prepare_scene_dir,
    prepare_traj_dir,
    read_json,
    write_traj_bundle,
)
from datagen.motion.action_space import Act, ActionSpec, actions_to_tokens, canonicalize, sample_magnitude
from datagen.motion.sampling import select_frame_indices
from datagen.motion.trajectory import Trajectory, build_trajectory
from datagen.quality.blur_detector import QualityAssessor
from datagen.quality.image_filter import ImageFilter, PlaceholderImageFilter
from datagen.quality.temporal_qc import check_temporal_smoothness


@dataclass
class TrajectoryRequest:
    traj_id: str
    kind: str
    name: str
    action_specs: list[ActionSpec]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified DataGen Framework")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--backend", type=str, default=None, choices=["svc", "worldfm"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--manifest_jsonl", type=str, default=None)
    parser.add_argument("--out_root", type=str, default=None)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--camera_config", type=str, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--num_chunks", type=int, default=None)
    parser.add_argument("--chunk_idx", type=int, default=None)
    parser.add_argument("--merge_chunks", action="store_true")
    return parser


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid config at {path}")
    return data


def _apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    if args.backend:
        out["model"]["backend"] = args.backend
    if args.device:
        out["general"]["device"] = args.device
    if args.input_dir:
        out["general"]["input_dir"] = args.input_dir
    if args.manifest_jsonl:
        out["general"]["manifest_jsonl"] = args.manifest_jsonl
    if args.out_root:
        out["output"]["root"] = args.out_root
    if args.version:
        out["general"]["version"] = args.version
    if args.seed is not None:
        out["general"]["seed"] = int(args.seed)
    if args.camera_config:
        out["general"]["camera_config"] = args.camera_config
    if args.skip_existing:
        out["general"]["skip_existing"] = True
    if args.num_chunks is not None:
        out["general"]["num_chunks"] = int(args.num_chunks)
    if args.chunk_idx is not None:
        out["general"]["chunk_idx"] = int(args.chunk_idx)
    if args.merge_chunks:
        out["general"]["merge_chunks"] = True
    return out


def _resolve_optional_path(raw_path: str, base_path: Path) -> Path:
    p = Path(raw_path).expanduser()
    if not p.is_absolute():
        cwd_candidate = (Path.cwd() / p).resolve()
        if cwd_candidate.exists():
            return cwd_candidate
        p = (base_path / p).resolve()
    return p


def _load_camera_config(cfg: dict[str, Any], cfg_path: Path) -> dict[str, Any] | None:
    raw_camera_cfg = str(cfg.get("general", {}).get("camera_config", "")).strip()
    if not raw_camera_cfg:
        return None
    camera_cfg_path = _resolve_optional_path(raw_camera_cfg, cfg_path.parent)
    camera_cfg_root = _load_yaml(camera_cfg_path)
    camera_cfg = camera_cfg_root.get("camera", camera_cfg_root)
    if not isinstance(camera_cfg, dict):
        raise RuntimeError(f"Invalid camera config at {camera_cfg_path}")
    return camera_cfg


def _scan_input_dir(input_dir: Path) -> list[dict[str, str]]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    images = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in exts]
    return [{"scene_id": p.stem, "anchor_path": str(p)} for p in images]


def _load_manifest(path: Path) -> list[dict[str, str]]:
    scenes = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            anchor_path = str(item["anchor_path"])
            # Always align scene folder name with image stem (user requirement).
            scene_id = Path(anchor_path).stem
            scenes.append({"scene_id": scene_id, "anchor_path": anchor_path})
    return scenes


def _write_manifest(path: Path, scenes: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for scene in scenes:
            f.write(json.dumps(scene, ensure_ascii=True))
            f.write("\n")
    tmp_path.replace(path)


def _build_rng(cfg: dict[str, Any]) -> np.random.Generator:
    seed_raw = cfg.get("general", {}).get("seed")
    # seed < 0 (or None) means random seed each run.
    if seed_raw is None:
        return np.random.default_rng()
    try:
        seed_val = int(seed_raw)
    except (TypeError, ValueError):
        return np.random.default_rng()
    if seed_val < 0:
        return np.random.default_rng()
    return np.random.default_rng(seed_val)


def _select_contiguous_chunk(
    scenes: list[dict[str, str]],
    num_chunks: int,
    chunk_idx: int,
) -> list[dict[str, str]]:
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    if chunk_idx < 0 or chunk_idx >= num_chunks:
        raise ValueError("chunk_idx must be within [0, num_chunks)")
    total = len(scenes)
    base = total // num_chunks
    extra = total % num_chunks
    start = chunk_idx * base + min(chunk_idx, extra)
    end = start + base + (1 if chunk_idx < extra else 0)
    return scenes[start:end]


def _chunk_output_paths(out_root: Path, num_chunks: int | None) -> tuple[list[Path], list[Path]]:
    if num_chunks is None:
        jsonl_files = sorted(out_root.glob("frames_chunk_*.jsonl"))
        stats_files = sorted(out_root.glob("stats_chunk_*.json"))
    else:
        jsonl_files = [out_root / f"frames_chunk_{idx}.jsonl" for idx in range(num_chunks)]
        stats_files = [out_root / f"stats_chunk_{idx}.json" for idx in range(num_chunks)]
    return jsonl_files, stats_files


def _write_empty_outputs(jsonl_path: Path, stats_path: Path) -> None:
    jsonl_path.write_text("", encoding="utf-8")
    stats_path.write_text(
        json.dumps(
            {"scenes": 0, "trajs": 0, "frames": 0, "qc_failed": 0, "gen_failed": 0},
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _merge_jsonl(files: list[Path], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as out:
        for fp in files:
            if not fp.exists():
                continue
            with fp.open("r", encoding="utf-8") as src:
                for line in src:
                    out.write(line)


def _merge_stats(files: list[Path], out_path: Path) -> None:
    merged = {"scenes": 0, "trajs": 0, "frames": 0, "qc_failed": 0, "gen_failed": 0}
    for fp in files:
        if not fp.exists():
            continue
        with fp.open("r", encoding="utf-8") as f:
            stats = json.load(f)
        for key in merged:
            merged[key] += int(stats.get(key, 0))
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=True, indent=2)


def _is_skippable_svc_generation_error(exc: AssertionError) -> bool:
    message = str(exc)
    return "More anchor frames need to be sampled during the first pass" in message


def _as_4x4(c2w: np.ndarray) -> np.ndarray:
    c2w_arr = np.asarray(c2w, dtype=np.float64)
    if c2w_arr.shape == (4, 4):
        return c2w_arr
    if c2w_arr.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = c2w_arr
        return out
    raise ValueError(f"c2w must be (3,4) or (4,4), got {c2w_arr.shape}")


def _build_transforms_payload(
    traj_dir: Path,
    frame_paths: list[Path],
    Ks: np.ndarray,
    c2ws: np.ndarray,
) -> dict[str, Any]:
    with Image.open(frame_paths[0]) as img:
        width, height = img.size
    frames: list[dict[str, Any]] = []
    for i, frame_path in enumerate(frame_paths):
        K = np.asarray(Ks[i], dtype=np.float64)
        c2w = _as_4x4(np.asarray(c2ws[i], dtype=np.float64))
        rel = str(Path("./") / frame_path.relative_to(traj_dir))
        frames.append(
            {
                "id": i + 1,
                "file_path": rel,
                "path": rel,
                "width": int(width),
                "height": int(height),
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
                "K": K.tolist(),
                "c2w": c2w.tolist(),
            }
        )
    return {"frames": frames}


def _sample_single_action(act_name: str, cfg: dict[str, Any], rng: np.random.Generator) -> ActionSpec:
    act = Act(act_name)
    if act in {Act.TURN_LEFT, Act.TURN_RIGHT}:
        angle_cfg = cfg["action"]["angle"]
        magnitude = sample_magnitude(
            rng,
            float(angle_cfg["min"]),
            float(angle_cfg["max"]),
            float(angle_cfg["unit"]),
        )
    else:
        dist_cfg = cfg["action"]["distance"]
        magnitude = sample_magnitude(
            rng,
            float(dist_cfg["min"]),
            float(dist_cfg["max"]),
            float(dist_cfg["unit"]),
        )
    return ActionSpec(act=act, magnitude=float(magnitude))


def _sample_multi_actions(preset: dict[str, Any], cfg: dict[str, Any], rng: np.random.Generator) -> list[ActionSpec]:
    actions: list[ActionSpec] = []
    for step in preset.get("steps", []):
        act = Act(str(step["act"]))
        if "distance_range" in step:
            min_val, max_val = step["distance_range"]
            unit = float(cfg["action"]["distance"]["unit"])
        elif "angle_range" in step:
            min_val, max_val = step["angle_range"]
            unit = float(cfg["action"]["angle"]["unit"])
        else:
            if act in {Act.TURN_LEFT, Act.TURN_RIGHT}:
                min_val = float(cfg["action"]["angle"]["min"])
                max_val = float(cfg["action"]["angle"]["max"])
                unit = float(cfg["action"]["angle"]["unit"])
            else:
                min_val = float(cfg["action"]["distance"]["min"])
                max_val = float(cfg["action"]["distance"]["max"])
                unit = float(cfg["action"]["distance"]["unit"])
        magnitude = sample_magnitude(rng, float(min_val), float(max_val), unit)
        actions.append(ActionSpec(act=act, magnitude=float(magnitude)))
    return actions


def _build_scene_requests(cfg: dict[str, Any], rng: np.random.Generator) -> list[TrajectoryRequest]:
    requests: list[TrajectoryRequest] = []
    for idx, name in enumerate(cfg["trajectory"]["single_step"]):
        action = _sample_single_action(str(name), cfg, rng)
        requests.append(
            TrajectoryRequest(
                traj_id=f"{name}_{idx:02d}",
                kind="single",
                name=str(name),
                action_specs=[action],
            )
        )
    for idx, preset in enumerate(cfg["trajectory"].get("multi_step_presets", [])):
        actions = _sample_multi_actions(preset, cfg, rng)
        preset_name = str(preset["name"])
        requests.append(
            TrajectoryRequest(
                traj_id=f"multi_step_{idx:02d}_{preset_name}",
                kind="multi",
                name=preset_name,
                action_specs=actions,
            )
        )
    return requests


def _materialize_trajectory(
    req: TrajectoryRequest,
    image_wh: tuple[int, int],
    cfg: dict[str, Any],
) -> Trajectory:
    if req.kind == "single":
        action = req.action_specs[0]
        return build_trajectory(
            req.name,
            traj_id=req.traj_id,
            image_wh=image_wh,
            magnitude=float(action.magnitude),
            cfg=cfg,
        )
    return build_trajectory(
        "multi_step",
        traj_id=req.traj_id,
        image_wh=image_wh,
        action_specs=req.action_specs,
        cfg=cfg,
    )


def _scaled_cfg_for_retry(cfg: dict[str, Any], scale: float) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out["action"]["distance"]["max_step_per_frame"] = float(cfg["action"]["distance"]["max_step_per_frame"]) / scale
    out["action"]["angle"]["max_step_per_frame"] = float(cfg["action"]["angle"]["max_step_per_frame"]) / scale
    return out


def _build_backend_options_record(cfg: dict[str, Any]) -> dict[str, Any]:
    backend = cfg["model"]["backend"]
    if backend == "svc":
        svc_cfg = cfg["svc"]
        return {
            "cfg": float(svc_cfg["cfg"]),
            "num_steps": int(svc_cfg["num_steps"]),
            "camera_scale": float(svc_cfg["camera_scale"]),
            "H": int(svc_cfg["H"]),
            "W": int(svc_cfg["W"]),
            "cfg_min": float(svc_cfg["cfg_min"]),
            "chunk_strategy": str(svc_cfg["chunk_strategy"]),
            "guider_types": list(svc_cfg["guider_types"]),
        }
    worldfm_cfg = cfg["worldfm"]
    return {
        "cfg": float(worldfm_cfg["cfg_scale"]),
        "num_steps": int(worldfm_cfg["step"]),
        "camera_scale": 1.0,
        "H": int(worldfm_cfg["image_size"]),
        "W": int(worldfm_cfg["image_size"]),
        "render_size": int(worldfm_cfg["render_size"]),
    }


def _resolve_runtime_paths(cfg: dict[str, Any]) -> tuple[Path, str | None, str | None]:
    out_root = Path(cfg["output"]["root"]).expanduser()
    version = str(cfg["general"].get("version", "")).strip()
    if version:
        out_root = out_root / version
    input_dir = str(cfg["general"].get("input_dir", "")).strip() or None
    manifest_jsonl = str(cfg["general"].get("manifest_jsonl", "")).strip() or None
    return out_root.resolve(), input_dir, manifest_jsonl


def _build_adapter(cfg: dict[str, Any]):
    backend = cfg["model"]["backend"]
    if backend == "svc":
        adapter = SVCAdapter()
        svc_cfg = copy.deepcopy(cfg["svc"])
        svc_cfg["device"] = cfg["general"]["device"]
        svc_cfg["fps"] = cfg["output"]["fps"]
        adapter.init_model(svc_cfg)
        return adapter
    if backend == "worldfm":
        adapter = WorldFMAdapter()
        worldfm_cfg = copy.deepcopy(cfg["worldfm"])
        worldfm_cfg["output_dir"] = str(Path(cfg["output"]["root"]).expanduser())
        adapter.init_model(worldfm_cfg)
        return adapter
    raise ValueError(f"Unsupported backend: {backend}")


def _process_scene(
    scene: dict[str, str],
    cfg: dict[str, Any],
    adapter: Any,
    quality_assessor: QualityAssessor,
    image_filter: ImageFilter,
    out_root: Path,
    jsonl_path: Path,
    rng: np.random.Generator,
) -> dict[str, int]:
    scene_id = scene["scene_id"]
    anchor_path = Path(scene["anchor_path"])
    scene_dir = prepare_scene_dir(out_root, scene_id)
    scene_input_dir = scene_dir / "input"
    scene_input_path = scene_input_dir / "000.png"
    if not scene_input_path.exists():
        scene_input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(anchor_path, scene_input_path)

    if cfg["quality"]["input_filter"]["enabled"] and not image_filter.is_suitable(str(scene_input_path)):
        return {"scenes": 1, "trajs": 0, "frames": 0, "qc_failed": 0, "gen_failed": 0}

    with Image.open(scene_input_path) as img:
        image_wh = img.size

    scene_ctx = adapter.prepare_scene(str(scene_input_path), str(scene_dir))
    requests = _build_scene_requests(cfg, rng)
    stats = {"scenes": 1, "trajs": 0, "frames": 0, "qc_failed": 0, "gen_failed": 0}
    backend_options_record_base = _build_backend_options_record(cfg)

    blur_cfg = cfg["quality"]["blur_detection"]
    blur_enabled = bool(blur_cfg["enabled"])
    max_retries = int(blur_cfg["max_retries"])
    frame_increase_factor = float(blur_cfg["frame_increase_factor"])
    temporal_cfg = cfg["quality"]["temporal_qc"]

    try:
        for req in requests:
            traj_dir = scene_dir / f"traj_{req.traj_id}"
            if cfg["general"]["skip_existing"] and (traj_dir / "traj.json").exists():
                continue
            if traj_dir.exists():
                shutil.rmtree(traj_dir, ignore_errors=True)
            traj_dir = prepare_traj_dir(scene_dir, req.traj_id)

            retry_idx = 0
            final_traj: Trajectory | None = None
            final_frames: list[Path] = []
            final_quality_scores: dict[str, Any] | None = None
            blur_triggered = False
            generation_failed = False

            while retry_idx <= max_retries:
                retry_scale = frame_increase_factor ** retry_idx
                retry_cfg = _scaled_cfg_for_retry(cfg, retry_scale)
                traj = _materialize_trajectory(req, image_wh, retry_cfg)
                options_override: dict[str, Any] = {
                    "write_mp4": bool(cfg["output"]["save_mp4"]),
                    "save_input": False,
                }
                if cfg["model"]["backend"] == "svc":
                    options_override["camera_scale"] = float(
                        traj.params.get("camera_scale", cfg["svc"].get("camera_scale", 2.0))
                    )
                try:
                    frame_paths = adapter.generate_frames(
                        image_path=str(scene_input_path),
                        c2ws=traj.c2ws,
                        Ks=traj.Ks,
                        save_dir=str(traj_dir),
                        scene_ctx=scene_ctx,
                        seed=int(rng.integers(1, 10_000_000)),
                        options_override=options_override,
                        write_mp4=bool(cfg["output"]["save_mp4"]),
                        fps=int(cfg["output"]["fps"]),
                    )
                except AssertionError as exc:
                    if not _is_skippable_svc_generation_error(exc):
                        raise
                    print(
                        f"[skip] scene={scene_id} traj={req.traj_id} dropped due to SVC chunking error: {exc}"
                    )
                    stats["gen_failed"] += 1
                    generation_failed = True
                    shutil.rmtree(traj_dir, ignore_errors=True)
                    break
                frames = [Path(p) for p in frame_paths]

                per_frame_quality: list[dict[str, Any]] = []
                is_low_quality = False
                for frame_idx, frame in enumerate(frames):
                    quality = quality_assessor.assess(str(frame))
                    per_frame_quality.append(
                        {
                            "frame_idx": int(frame_idx),
                            "laplacian_var": float(quality.laplacian_var),
                            "brisque": (
                                None if quality.brisque_score is None else float(quality.brisque_score)
                            ),
                            "is_blurry": bool(quality.is_blurry),
                            "is_low_quality": bool(quality.is_low_quality),
                        }
                    )
                    is_low_quality = is_low_quality or bool(quality.is_low_quality)

                laplacian_values = [float(item["laplacian_var"]) for item in per_frame_quality]
                brisque_values = [float(item["brisque"]) for item in per_frame_quality if item["brisque"] is not None]
                quality_scores = {
                    "per_frame": per_frame_quality,
                    "min_laplacian_var": (min(laplacian_values) if laplacian_values else None),
                    "max_laplacian_var": (max(laplacian_values) if laplacian_values else None),
                    "mean_laplacian_var": (sum(laplacian_values) / len(laplacian_values) if laplacian_values else None),
                    "mean_brisque": (sum(brisque_values) / len(brisque_values) if brisque_values else None),
                    "passed": (not is_low_quality),
                }

                final_traj = traj
                final_frames = frames
                final_quality_scores = quality_scores
                if blur_enabled and is_low_quality and retry_idx < max_retries:
                    blur_triggered = True
                    retry_idx += 1
                    shutil.rmtree(traj_dir, ignore_errors=True)
                    traj_dir = prepare_traj_dir(scene_dir, req.traj_id)
                    continue
                blur_triggered = blur_triggered or (blur_enabled and is_low_quality)
                break

            if generation_failed:
                continue

            if final_traj is None or not final_frames:
                stats["qc_failed"] += 1
                continue

            qc_passed = True
            qc_info: dict[str, Any] = {
                "passed": True,
                "filters": {
                    "blur_retry": {
                        "triggered": blur_triggered,
                        "retry_count": retry_idx,
                        "laplacian_threshold": float(blur_cfg.get("laplacian_threshold", 100.0)),
                        "brisque_threshold": float(blur_cfg.get("brisque_threshold", 50.0)),
                    }
                },
            }
            if final_quality_scores is not None:
                qc_info["filters"]["quality_scores"] = {
                    "passed": bool(final_quality_scores.get("passed", True)),
                    "min_laplacian_var": final_quality_scores.get("min_laplacian_var"),
                    "mean_brisque": final_quality_scores.get("mean_brisque"),
                }
                if blur_enabled and not bool(final_quality_scores.get("passed", True)):
                    qc_passed = False
                    qc_info["passed"] = False
            if temporal_cfg["enabled"]:
                passed, metrics = check_temporal_smoothness(
                    final_frames,
                    min_diff=float(temporal_cfg["min_diff"]),
                    max_diff=float(temporal_cfg["max_diff"]),
                )
                qc_passed = passed
                qc_info["passed"] = passed
                qc_info["filters"]["temporal"] = metrics

            if not qc_passed:
                stats["qc_failed"] += 1
                shutil.rmtree(traj_dir, ignore_errors=True)
                continue

            keep_ids = select_frame_indices(
                num_frames=len(final_frames),
                keep_k=int(cfg["sampling"]["frame_keep_k"]),
                is_orbit=False,
            )

            transforms_path = traj_dir / "transforms.json"
            if transforms_path.exists():
                transforms_payload = read_json(transforms_path)
            else:
                transforms_payload = _build_transforms_payload(traj_dir, final_frames, final_traj.Ks, final_traj.c2ws)

            write_traj_bundle(
                traj_dir=traj_dir,
                traj_params=final_traj.params,
                actions_step=final_traj.actions_step,
                actions_prefix=final_traj.actions_prefix,
                frame_ids_kept=keep_ids,
                transforms=transforms_payload,
                quality_scores=final_quality_scores,
                key_frame_indices=final_traj.key_frame_indices,
            )

            # Keep only one anchor input per scene.
            traj_input_dir = traj_dir / "input"
            traj_input_path = traj_input_dir / "000.png"
            if not scene_input_path.exists() and traj_input_path.exists():
                scene_input_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(traj_input_path), str(scene_input_path))
            shutil.rmtree(traj_input_dir, ignore_errors=True)

            # Remove redundant intermediate files.
            for redundant_name in ("input.mp4", "samples-rgb.mp4", "output.mp4"):
                redundant_path = traj_dir / redundant_name
                if redundant_path.exists() and (redundant_name != "output.mp4" or not cfg["output"]["save_mp4"]):
                    redundant_path.unlink()
            shutil.rmtree(traj_dir / "first-pass", ignore_errors=True)
            shutil.rmtree(traj_dir / "second-pass", ignore_errors=True)
            if transforms_path.exists():
                transforms_path.unlink()

            prev_kept = None
            for frame_idx in keep_ids:
                if prev_kept is None:
                    step_actions: list[ActionSpec] = []
                else:
                    step_actions = merge_actions_between(final_traj.actions_step, prev_kept, frame_idx)
                prefix_actions = final_traj.actions_prefix[frame_idx]
                record = frame_record(
                    uid=f"{scene_id}_{final_traj.traj_id}_f{frame_idx:03d}",
                    scene_id=scene_id,
                    anchor_path=str(scene_input_path),
                    image_path=str(final_frames[frame_idx]),
                    traj_id=final_traj.traj_id,
                    traj_params=final_traj.params,
                    frame_idx=frame_idx,
                    actions_prefix=actions_to_tokens(prefix_actions),
                    actions_step=actions_to_tokens(canonicalize(step_actions)),
                    svc_options={
                        **backend_options_record_base,
                        **(
                            {"camera_scale": float(final_traj.params.get("camera_scale"))}
                            if "camera_scale" in final_traj.params
                            else {}
                        ),
                    },
                    qc=qc_info,
                )
                append_jsonl(jsonl_path, record)
                prev_kept = frame_idx
                stats["frames"] += 1
            stats["trajs"] += 1
    finally:
        adapter.release_scene(scene_ctx)
    return stats


def run_dataset(args: argparse.Namespace) -> None:
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(cfg_path)
    cfg = _apply_cli_overrides(cfg, args)
    camera_cfg = _load_camera_config(cfg, cfg_path)
    if camera_cfg is not None:
        cfg["camera"] = camera_cfg
    out_root, input_dir, manifest_jsonl = _resolve_runtime_paths(cfg)
    out_root.mkdir(parents=True, exist_ok=True)

    num_chunks = int(cfg["general"].get("num_chunks", 0))
    chunk_idx = int(cfg["general"].get("chunk_idx", 0))
    merge_chunks = bool(cfg["general"].get("merge_chunks", False))

    if merge_chunks:
        jsonl_files, stats_files = _chunk_output_paths(out_root, num_chunks if num_chunks > 0 else None)
        if not any(fp.exists() for fp in jsonl_files + stats_files):
            raise RuntimeError("No chunk outputs found to merge.")
        _merge_jsonl(jsonl_files, out_root / "frames.jsonl")
        _merge_stats(stats_files, out_root / "stats.json")
        return

    scenes: list[dict[str, str]] = []
    if manifest_jsonl:
        scenes = _load_manifest(Path(manifest_jsonl).expanduser().resolve())
    elif input_dir:
        input_dir_path = Path(input_dir).expanduser().resolve()
        manifest_path = out_root / "manifest.jsonl"
        if manifest_path.exists():
            scenes = _load_manifest(manifest_path)
        else:
            scenes = _scan_input_dir(input_dir_path)
            if scenes:
                _write_manifest(manifest_path, scenes)
                scenes = _load_manifest(manifest_path)
    if not scenes:
        raise RuntimeError(
            "No input images found. "
            f"manifest_jsonl={manifest_jsonl!r}, input_dir={input_dir!r}, out_root={str(out_root)!r}"
        )

    if num_chunks > 0:
        scenes = _select_contiguous_chunk(scenes, num_chunks, chunk_idx)
        jsonl_path = out_root / f"frames_chunk_{chunk_idx}.jsonl"
        stats_path = out_root / f"stats_chunk_{chunk_idx}.json"
        if not scenes:
            _write_empty_outputs(jsonl_path, stats_path)
            return
    else:
        jsonl_path = out_root / "frames.jsonl"
        stats_path = out_root / "stats.json"

    if jsonl_path.exists():
        jsonl_path.unlink()

    rng = _build_rng(cfg)
    adapter = _build_adapter(cfg)
    blur_cfg = cfg["quality"]["blur_detection"]
    quality_assessor = QualityAssessor(
        laplacian_threshold=float(blur_cfg.get("laplacian_threshold", 100.0)),
        brisque_threshold=float(blur_cfg.get("brisque_threshold", 50.0)),
        brisque_model_path=str(blur_cfg.get("brisque_model_path", "")),
        brisque_range_path=str(blur_cfg.get("brisque_range_path", "")),
    )
    image_filter: ImageFilter = PlaceholderImageFilter()

    stats = {"scenes": 0, "trajs": 0, "frames": 0, "qc_failed": 0, "gen_failed": 0}
    try:
        for scene in scenes:
            scene_stats = _process_scene(
                scene=scene,
                cfg=cfg,
                adapter=adapter,
                quality_assessor=quality_assessor,
                image_filter=image_filter,
                out_root=out_root,
                jsonl_path=jsonl_path,
                rng=rng,
            )
            for key, value in scene_stats.items():
                stats[key] += int(value)
    finally:
        adapter.close()

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=True, indent=2)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_dataset(args)


if __name__ == "__main__":
    main()
