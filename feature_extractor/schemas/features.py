from typing import Any, Literal
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class PreparedImage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Image.Image          # RGB, exactly size x size
    original_hw: list[int]      # after EXIF transpose, before resize
    stored_hw: list[int]        # as stored in the file, before EXIF transpose
    exif_orientation: int


class ExtractedFeatures(BaseModel):
    """Raw model output for one image, before serialization."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    feature_vector: np.ndarray                  # (hidden * len(layers),) float32, L2-normalized
    feature_maps: dict[int, np.ndarray]         # layer -> (hidden, grid_h, grid_w) float32
    image_meta: dict[str, Any] = {}             # input_hw, grid_hw, ... from the model side


class ExtractFeaturesRequest(BaseModel):
    include_feature_map: bool = Field(
        default=False,
        description="Set true to also return the per-layer feature maps (about 2.8 MB of base64 per layer)"
    )


class ArrayPayload(BaseModel):
    name: str
    shape: list[int]
    dtype: str
    encoding: Literal["list", "base64"]
    data: list[float] | str = Field(
        description="list: plain JSON floats. "
                    "base64: little-endian raw bytes of `dtype` in C order; reshape to `shape`"
    )


class FeatureMapPayload(ArrayPayload):
    layer: int


class ImageInfo(BaseModel):
    filename: str | None
    content_type: str | None
    stored_hw: list[int]
    original_hw: list[int]
    exif_orientation: int
    input_hw: list[int]
    grid_hw: list[int]


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    checkpoint: str
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    patch_size: list[int]
    num_register_tokens: int
    layers: list[int]
    device: str
    effective_amp: str


class ExtractFeaturesResponse(BaseModel):
    image: ImageInfo
    model: ModelInfo
    storage_dtype: str
    feature_vector: ArrayPayload
    feature_maps: list[FeatureMapPayload]
