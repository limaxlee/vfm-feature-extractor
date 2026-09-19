import os
import sys
import logging
import threading
from typing import Any

from PIL import Image

from common.config import ModelConfig
from common.constants import TORCH_ENV, REQUESTED_FEATURE_NAMES, ExtractorKeys
from feature_extractor.schemas import ExtractedFeatures

# Allocator settings must be in the environment before torch is first imported.
for _name, _value in TORCH_ENV.items():
    os.environ.setdefault(_name, _value)

logger = logging.getLogger(__name__)


class VFMExtractor:
    """One resident DinoFeatureExtractor from cvpr27/platform/feature_export/extract.py.

    Inference is serialized with a lock: one GPU, one model, batch size 1.
    """

    def __init__(self, config: ModelConfig, image_size: int):
        platform_dir = os.path.abspath(os.path.join(config.root, "platform"))
        if not os.path.isdir(os.path.join(platform_dir, "feature_export")):
            raise FileNotFoundError(f"feature_export package not found under {platform_dir}")
        if platform_dir not in sys.path:
            sys.path.insert(0, platform_dir)

        from feature_export.extract import DinoFeatureExtractor

        checkpoint_path = config.checkpoint
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(config.root, checkpoint_path)

        logger.info(f"Loading DINOv3 checkpoint {checkpoint_path} on {config.device} with amp {config.amp}")
        self._model = DinoFeatureExtractor(
            checkpoint_path,
            layers=tuple(config.layers),
            device=config.device,
            amp=config.amp,
            l2_patches=config.l2_patches,
        )
        self._input_size = (image_size, image_size)
        self._lock = threading.Lock()
        self._info = self._model.describe()
        logger.info(
            f"Model ready: hidden {self._info['hidden_size']}, layers {self._info['layers']}, "
            f"patch {self._info['patch_size']}"
        )

    @property
    def info(self) -> dict[str, Any]:
        return dict(self._info)

    @property
    def layers(self) -> list[int]:
        return list(self._model.layers)

    def warmup(self) -> None:
        self.extract(Image.new("RGB", self._input_size))

    def extract(self, image: Image.Image) -> ExtractedFeatures:
        with self._lock:
            # extract_images is a generator; consume it fully so nothing stays suspended holding tensors.
            records = list(self._model.extract_images(
                [image],
                input_size=self._input_size,
                resize_mode="resize",
                batch_size=1,
                feature_names=REQUESTED_FEATURE_NAMES,
            ))

        features = records[0]["features"]
        record = records[0]
        vector = features[ExtractorKeys.FEATURE_VECTOR].numpy()
        feature_maps = {
            layer: features[f"layer_{layer}_{ExtractorKeys.FEATURE_MAP}"].numpy()
            for layer in self._model.layers
        }
        return ExtractedFeatures(feature_vector=vector, feature_maps=feature_maps, image_meta=record["metadata"])

    def close(self) -> None:
        import torch

        with self._lock:
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
