import json
import base64
from io import BytesIO
from typing import Any

import numpy as np

from common.constants import ExtractorKeys


def convert_to_storage_dtype(array: np.ndarray, storage_dtype: str) -> np.ndarray:
    with np.errstate(over="ignore"):
        converted = np.ascontiguousarray(array, dtype=np.float32).astype(storage_dtype)
    if not np.isfinite(converted).all():
        raise ValueError(f"non-finite or overflowed values after casting to {storage_dtype}")
    return converted


def encode_array_to_base64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def decode_base64_to_array(data: str, shape: list[int], dtype: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(data), dtype=dtype).reshape(shape)


def pack_arrays_to_npz(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> bytes:
    """Same layout as the .npz written by extract.py: arrays + a metadata_json unicode scalar."""
    metadata = {**metadata, "shapes": {key: list(value.shape) for key, value in arrays.items()}}
    payload = dict(arrays)
    payload[ExtractorKeys.METADATA] = np.asarray(json.dumps(metadata, ensure_ascii=False, allow_nan=False))

    with BytesIO() as buffer:
        np.savez(buffer, **payload)
        return buffer.getvalue()
