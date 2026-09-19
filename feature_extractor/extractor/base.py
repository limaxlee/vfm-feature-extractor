from typing import Any, Protocol
from PIL import Image

from feature_extractor.schemas import ExtractedFeatures


class FeatureExtractor(Protocol):
    @property
    def info(self) -> dict[str, Any]:
        """Model description (checkpoint, layers, hidden_size, patch_size, device, ...)."""

    @property
    def layers(self) -> list[int]:
        ...

    def warmup(self) -> None:
        """Run one dummy forward so the first real request is not slow."""

    def extract(self, image: Image.Image) -> ExtractedFeatures:
        """Extract features from an RGB image already resized to the configured size."""

    def close(self) -> None:
        ...
