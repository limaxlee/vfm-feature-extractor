import posixpath
import hashlib
from typing import Any

import numpy as np
from PIL import Image

from common.config import ModelConfig
from common.constants import DEFAULT_PATCH_SIZE, HIDDEN_SIZE
from feature_extractor.schemas import ExtractedFeatures


class FakeExtractor:
    """Deterministic stand-in with the real output shapes. Never imports torch."""

    def __init__(self, config: ModelConfig, image_size: int):
        self._layers = list(config.layers)
        self._image_size = image_size
        self._grid = image_size // DEFAULT_PATCH_SIZE
        self._info = {
            "checkpoint": posixpath.join(config.root, config.checkpoint),
            "model_type": "fake",
            "hidden_size": HIDDEN_SIZE,
            "num_hidden_layers": 24,
            "patch_size": [DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE],
            "num_register_tokens": 4,
            "layers": self._layers,
            "device": "cpu",
            "effective_amp": "fp32",
        }

    @property
    def info(self) -> dict[str, Any]:
        return dict(self._info)

    @property
    def layers(self) -> list[int]:
        return list(self._layers)

    def warmup(self) -> None:
        self.extract(Image.new("RGB", (self._image_size, self._image_size)))

    def extract(self, image: Image.Image) -> ExtractedFeatures:
        if image.mode != "RGB" or image.size != (self._image_size, self._image_size):
            raise ValueError(f"expected an RGB image of size {self._image_size}x{self._image_size}")

        seed = int.from_bytes(hashlib.sha256(image.tobytes()).digest()[:8], "little")
        rng = np.random.default_rng(seed)

        feature_maps, means = {}, []
        for layer in self._layers:
            dense = rng.standard_normal((HIDDEN_SIZE, self._grid, self._grid), dtype=np.float32)
            feature_maps[layer] = dense
            mean = dense.reshape(HIDDEN_SIZE, -1).mean(axis=1)
            means.append(mean / np.linalg.norm(mean))

        vector = np.concatenate(means).astype(np.float32)
        vector /= np.linalg.norm(vector)

        meta = {
            "input_hw": [self._image_size, self._image_size],
            "grid_hw": [self._grid, self._grid],
            "resize_mode": "resize",
        }
        return ExtractedFeatures(feature_vector=vector, feature_maps=feature_maps, image_meta=meta)

    def close(self) -> None:
        return
