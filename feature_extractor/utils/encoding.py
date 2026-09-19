import base64

import numpy as np


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
