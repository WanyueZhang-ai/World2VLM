from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from datagen.adapters.base import ModelAdapter


def _ensure_homogeneous(c2ws: np.ndarray) -> np.ndarray:
    if c2ws.shape[-2:] == (4, 4):
        return c2ws
    if c2ws.shape[-2:] != (3, 4):
        raise ValueError("c2ws must have shape [N,3,4] or [N,4,4].")
    bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=c2ws.dtype)[None, None, :]
    bottom = np.repeat(bottom, c2ws.shape[0], axis=0)
    return np.concatenate([c2ws, bottom], axis=1)


class SVCAdapter(ModelAdapter):
    def __init__(self) -> None:
        self.device = "cuda:0"
        self.cfg: dict[str, Any] = {}
        self.model = None
        self.ae = None
        self.conditioner = None
        self.denoiser = None

    def init_model(self, cfg: dict[str, Any]) -> None:
        import torch
        from seva.model import SGMWrapper
        from seva.modules.autoencoder import AutoEncoder
        from seva.modules.conditioner import CLIPConditioner
        from seva.sampling import DiscreteDenoiser
        from seva.utils import load_model

        self.cfg = dict(cfg)
        self.device = str(cfg.get("device", self.device))
        pretrained_model_name_or_path = str(cfg["pretrained_model_name_or_path"])
        weight_name = str(cfg.get("weight_name", "model.safetensors"))
        compile_model = bool(cfg.get("compile_model", False))

        self.model = SGMWrapper(
            load_model(
                model_version=1.1,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                weight_name=weight_name,
                device="cpu",
                verbose=True,
            ).eval()
        ).to(self.device)
        self.ae = AutoEncoder(chunk_size=1).to(self.device)
        self.conditioner = CLIPConditioner().to(self.device)
        self.denoiser = DiscreteDenoiser(num_idx=1000, device=self.device)

        if compile_model:
            self.model = torch.compile(self.model, dynamic=False)
            self.ae = torch.compile(self.ae, dynamic=False)
            self.conditioner = torch.compile(self.conditioner, dynamic=False)

    def _build_version_dict(self, T: int, options_override: dict[str, Any] | None) -> dict[str, Any]:
        version_dict = {
            "H": int(self.cfg.get("H", 576)),
            "W": int(self.cfg.get("W", 576)),
            "T": T,
            "C": 4,
            "f": 8,
            "options": {
                "chunk_strategy": str(self.cfg.get("chunk_strategy", "interp")),
                "video_save_fps": float(self.cfg.get("fps", 30.0)),
                "beta_linear_start": 5e-6,
                "log_snr_shift": 2.4,
                "guider_types": list(self.cfg.get("guider_types", [1, 2])),
                "cfg": float(self.cfg.get("cfg", 4.0)),
                "camera_scale": float(self.cfg.get("camera_scale", 2.0)),
                "num_steps": int(self.cfg.get("num_steps", 50)),
                "cfg_min": float(self.cfg.get("cfg_min", 1.2)),
                "encoding_t": 1,
                "decoding_t": 1,
                "replace_or_include_input": True,
                "save_input": False,
                "write_transforms": False,
                "write_mp4": False,
                "save_first_pass": False,
                "save_second_pass": False,
            },
        }
        if options_override:
            version_dict["options"].update(options_override)
        return version_dict

    def _infer_prior_indices(self, version_dict: dict[str, Any], num_targets: int) -> list[int]:
        from seva.eval import infer_prior_stats

        # For very short trajectories, prior estimation in seva may divide by zero.
        if num_targets <= 1:
            return []
        local = copy.deepcopy(version_dict)
        num_anchors = infer_prior_stats(
            local["T"],
            num_input_frames=1,
            num_total_frames=num_targets,
            version_dict=local,
        )
        if num_anchors <= 0:
            return []
        anchor_indices = np.linspace(1, num_targets, num_anchors).tolist()
        return [int(round(i)) for i in anchor_indices]

    def prepare_scene(self, image_path: str, scene_dir: str) -> None:
        _ = image_path, scene_dir
        return None

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
        from seva.eval import create_transforms_simple, run_one_scene

        _ = scene_ctx
        if self.model is None or self.ae is None or self.conditioner is None or self.denoiser is None:
            raise RuntimeError("SVCAdapter is not initialized. Call init_model() first.")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        num_frames = int(c2ws.shape[0])

        options_override = kwargs.get("options_override", None)
        version_dict = self._build_version_dict(num_frames, options_override)
        options = version_dict["options"]
        options["num_targets"] = max(0, num_frames - 1)

        # Keep camera conditioning on CPU; seva.eval moves tensors to CUDA later.
        # This avoids device-mismatch in internal camera normalization logic.
        c2ws_th = torch.as_tensor(c2ws, dtype=torch.float32)
        Ks_th = torch.as_tensor(Ks, dtype=torch.float32)

        image_cond: dict[str, Any] = {
            "img": [str(image_path)] + [None] * (num_frames - 1),
            "input_indices": [0],
            "prior_indices": None,
        }

        use_traj_prior = bool(options.get("use_traj_prior", True))
        traj_prior_c2ws = None
        traj_prior_Ks = None
        if use_traj_prior:
            prior_indices = self._infer_prior_indices(version_dict, num_frames - 1)
            image_cond["prior_indices"] = prior_indices
            if prior_indices:
                traj_prior_c2ws = c2ws_th[prior_indices].clone()
                traj_prior_Ks = Ks_th[prior_indices].clone()
            else:
                use_traj_prior = False
                image_cond["prior_indices"] = None

        camera_cond = {
            "c2w": c2ws_th.clone(),
            "K": Ks_th.clone(),
            "input_indices": list(range(num_frames)),
        }

        version_dict_run = copy.deepcopy(version_dict)
        video_path_generator = run_one_scene(
            "img2trajvid_s-prob",
            version_dict_run,
            model=self.model,
            ae=self.ae,
            conditioner=self.conditioner,
            denoiser=self.denoiser,
            image_cond=image_cond,
            camera_cond=camera_cond,
            save_path=str(save_path),
            use_traj_prior=use_traj_prior,
            traj_prior_Ks=traj_prior_Ks,
            traj_prior_c2ws=traj_prior_c2ws,
            seed=int(seed),
        )
        for _ in video_path_generator:
            pass

        img_paths = sorted((save_path / "samples-rgb").glob("*.png"))

        if options.get("write_transforms", True) and img_paths:
            c2ws_h = _ensure_homogeneous(c2ws)
            c2ws_gl = c2ws_h @ np.diag([1.0, -1.0, -1.0, 1.0])
            img_wh = np.array(
                [[int(version_dict_run["W"]), int(version_dict_run["H"])]],
                dtype=np.int32,
            )
            img_whs = np.repeat(img_wh, num_frames, axis=0)
            create_transforms_simple(
                save_path=str(save_path),
                img_paths=[str(p) for p in img_paths],
                img_whs=img_whs,
                c2ws=c2ws_gl,
                Ks=Ks,
            )

        return [str(p) for p in img_paths]

    def release_scene(self, scene_ctx: Any) -> None:
        _ = scene_ctx

    def close(self) -> None:
        import gc

        try:
            import torch
        except Exception:
            torch = None

        if hasattr(self, "model"):
            self.model = None
        if hasattr(self, "ae"):
            self.ae = None
        if hasattr(self, "conditioner"):
            self.conditioner = None
        if hasattr(self, "denoiser"):
            self.denoiser = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

