from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ModelAdapter(ABC):
    @abstractmethod
    def init_model(self, cfg: dict[str, Any]) -> None:
        """Initialize model resources."""

    @abstractmethod
    def prepare_scene(self, image_path: str, scene_dir: str) -> Any:
        """Prepare scene-level context. SVC may return None."""

    @abstractmethod
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
        """Generate frames and return file paths."""

    @abstractmethod
    def release_scene(self, scene_ctx: Any) -> None:
        """Release per-scene resources."""

    @abstractmethod
    def close(self) -> None:
        """Release global model resources."""

