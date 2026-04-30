from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from datagen.adapters.base import ModelAdapter


@dataclass
class SceneBundle:
    scene_id: str
    scene_dir: Path
    image_path: Path
    pp_result: Any
    renderer: Any
    cond_db: Any
    rcfg: Any
    render_size: int


def _as_4x4(c2w: np.ndarray) -> np.ndarray:
    c2w_arr = np.asarray(c2w, dtype=np.float64)
    if c2w_arr.shape == (4, 4):
        return c2w_arr
    if c2w_arr.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = c2w_arr
        return out
    raise ValueError(f"c2w must be (3,4) or (4,4), got {c2w_arr.shape}")


def _write_video(video_path: Path, frames: list[np.ndarray], fps: int) -> None:
    import cv2

    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


class WorldFMAdapter(ModelAdapter):
    def __init__(self) -> None:
        self.cfg: dict[str, Any] = {}
        self.pipeline_cfg: Any = None
        self.svc: Any = None
        self.wcfg: Any = None
        self.worldfm_pipeline: Any = None

    def init_model(self, cfg: dict[str, Any]) -> None:
        import torch
        from omegaconf import OmegaConf

        import run_pipeline as worldfm_pipeline

        self.cfg = dict(cfg)
        self.worldfm_pipeline = worldfm_pipeline
        gpu_index = int(cfg.get("gpu_index", 0))
        if gpu_index >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(gpu_index)

        default_cfg_dict = OmegaConf.to_container(worldfm_pipeline.DEFAULT_CFG, resolve=False)
        merged = OmegaConf.create(default_cfg_dict)
        config_path = str(cfg.get("config_path", "")).strip()
        if config_path:
            user_cfg = OmegaConf.load(config_path)
            merged = OmegaConf.merge(merged, user_cfg)

        cli_overrides = OmegaConf.create(
            {
                "pipeline": {
                    "output_dir": str(cfg.get("output_dir", "outputs")),
                    "gpu_index": gpu_index,
                },
                "submodules": {
                    "hw_path": str(cfg.get("hw_path", merged.submodules.hw_path)),
                    "moge_path": str(cfg.get("moge_path", merged.submodules.moge_path)),
                },
                "moge": {
                    "pretrained": str(cfg.get("moge_pretrained", merged.moge.pretrained)),
                },
                "render": {
                    "render_size": int(cfg.get("render_size", merged.render.render_size)),
                },
                "worldfm": {
                    "model_path": str(cfg.get("model_path", merged.worldfm.model_path)),
                    "vae_path": str(cfg.get("vae_path", merged.worldfm.vae_path)),
                    "image_size": int(cfg.get("image_size", merged.worldfm.image_size)),
                    "step": int(cfg.get("step", merged.worldfm.step)),
                    "cfg_scale": float(cfg.get("cfg_scale", merged.worldfm.cfg_scale)),
                },
            }
        )
        self.pipeline_cfg = OmegaConf.merge(merged, cli_overrides)

        worldfm_pipeline.setup_external_repos(
            hw_path=str(self.pipeline_cfg.submodules.hw_path),
            moge_path=str(self.pipeline_cfg.submodules.moge_path),
        )
        self.svc, self.wcfg = worldfm_pipeline.step4_init(cfg=self.pipeline_cfg)

    def prepare_scene(self, image_path: str, scene_dir: str) -> SceneBundle:
        if self.worldfm_pipeline is None or self.pipeline_cfg is None:
            raise RuntimeError("WorldFMAdapter is not initialized. Call init_model() first.")

        image_path_p = Path(image_path)
        scene_dir_p = Path(scene_dir)
        scene_dir_p.mkdir(parents=True, exist_ok=True)

        panorama_img = self.worldfm_pipeline.step1_panogen(
            image_path=str(image_path_p),
            output_dir=scene_dir_p,
            cfg=self.pipeline_cfg,
        )
        pp_result = self.worldfm_pipeline.step2_moge_pipeline(
            panorama_img=panorama_img,
            output_dir=scene_dir_p,
            cfg=self.pipeline_cfg,
            pretrained=str(self.pipeline_cfg.moge.pretrained),
        )
        renderer, cond_db, rcfg, render_size = self.worldfm_pipeline.step3_init(
            pp_result=pp_result,
            cfg=self.pipeline_cfg,
            render_size=int(self.pipeline_cfg.render.render_size),
        )
        return SceneBundle(
            scene_id=scene_dir_p.name,
            scene_dir=scene_dir_p,
            image_path=image_path_p,
            pp_result=pp_result,
            renderer=renderer,
            cond_db=cond_db,
            rcfg=rcfg,
            render_size=int(render_size),
        )

    def generate_frames(
        self,
        image_path: str,
        c2ws: np.ndarray,
        Ks: np.ndarray,
        save_dir: str,
        scene_ctx: Any = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> list[str]:
        import torch
        from PIL import Image

        _ = image_path
        if self.worldfm_pipeline is None or self.svc is None or self.wcfg is None:
            raise RuntimeError("WorldFMAdapter is not initialized. Call init_model() first.")
        if scene_ctx is None:
            raise ValueError("WorldFMAdapter.generate_frames requires scene_ctx from prepare_scene.")

        save_dir_p = Path(save_dir)
        samples_dir = save_dir_p / "samples-rgb"
        samples_dir.mkdir(parents=True, exist_ok=True)

        frame_paths: list[str] = []
        frame_arrays: list[np.ndarray] = []
        for frame_idx in range(int(c2ws.shape[0])):
            frame_seed = int(seed) + frame_idx
            torch.manual_seed(frame_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(frame_seed)

            K = np.asarray(Ks[frame_idx], dtype=np.float64)
            c2w = _as_4x4(np.asarray(c2ws[frame_idx], dtype=np.float64))
            render_u8, cond_nearest = self.worldfm_pipeline.step3_render_one(
                scene_ctx.renderer,
                scene_ctx.cond_db,
                scene_ctx.pp_result,
                K,
                c2w,
                rcfg=scene_ctx.rcfg,
                render_size=scene_ctx.render_size,
            )
            frame = self.worldfm_pipeline.step4_infer_one(
                self.svc,
                render_u8,
                cond_nearest,
                wcfg=self.wcfg,
            )
            frame_path = samples_dir / f"{frame_idx:04d}.png"
            Image.fromarray(frame, mode="RGB").save(frame_path)
            frame_paths.append(str(frame_path))
            frame_arrays.append(frame)

        if kwargs.get("write_mp4", True):
            fps = int(kwargs.get("fps", 30))
            _write_video(save_dir_p / "output.mp4", frame_arrays, fps)

        return frame_paths

    def release_scene(self, scene_ctx: Any) -> None:
        try:
            import torch
        except Exception:
            torch = None

        if scene_ctx is not None:
            if hasattr(scene_ctx, "renderer"):
                scene_ctx.renderer = None
            if hasattr(scene_ctx, "cond_db"):
                scene_ctx.cond_db = None
            if hasattr(scene_ctx, "pp_result"):
                scene_ctx.pp_result = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self) -> None:
        try:
            import torch
        except Exception:
            torch = None

        if hasattr(self, "svc"):
            self.svc = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

