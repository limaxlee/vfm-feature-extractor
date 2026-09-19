import os
from enum import StrEnum

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SERVICE_NAME = "VFM Feature Extractor"
SERVICE_VERSION = "0.1.0"

# Every input image is resized to IMAGE_SIZE x IMAGE_SIZE before extraction.
IMAGE_SIZE = 512

# Model defaults, matching the reference command for platform/feature_export/extract.py.
DEFAULT_LAYERS = (20, 24)
DEFAULT_PATCH_SIZE = 16
HIDDEN_SIZE = 1024


class ExtractorKeys(StrEnum):
    """Keys as produced by DinoFeatureExtractor.extract_batch and written by extract.py."""
    FEATURE_VECTOR = "patch_mean_concat"
    FEATURE_MAP = "feature_map"         # suffix: layer_{L}_feature_map, shape (hidden, grid_h, grid_w)


# feature_names passed to extract_images so unrequested dense outputs are never moved to CPU.
REQUESTED_FEATURE_NAMES = (ExtractorKeys.FEATURE_MAP, ExtractorKeys.FEATURE_VECTOR)


# Upload validation. Missing / octet-stream content types are allowed and validated by decoding.
SUPPORTED_CONTENT_TYPES = frozenset({
    "image/jpeg", "image/png", "image/bmp", "image/webp", "image/tiff", "application/octet-stream"
})
# PIL modes accepted by preprocess_image in platform/feature_export/extract.py.
SUPPORTED_IMAGE_MODES = frozenset({"1", "L", "LA", "P", "RGB", "RGBA", "CMYK"})
UPLOAD_CHUNK_BYTES = 1024 * 1024

# Must be set BEFORE torch is imported. Applied with os.environ.setdefault in
# feature_extractor/extractor/vfm.py and exported again by run.sh.
TORCH_ENV = {
    "PYTORCH_ALLOC_CONF": "backend:native,expandable_segments:False",
    "PYTORCH_CUDA_ALLOC_CONF": "backend:native,expandable_segments:False",
    "TORCH_SHOW_CPP_STACKTRACES": "1",
}
