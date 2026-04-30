from __future__ import annotations

from abc import ABC, abstractmethod


class ImageFilter(ABC):
    @abstractmethod
    def is_suitable(self, image_path: str) -> bool:
        """Return whether input image is suitable for generation."""


class PlaceholderImageFilter(ImageFilter):
    """Placeholder implementation before filter rules are finalized."""

    def is_suitable(self, image_path: str) -> bool:
        _ = image_path
        return True

