import numpy as np
import pytest

from feature_extractor.utils import convert_to_storage_dtype, encode_array_to_base64, decode_base64_to_array


def test_convert_to_storage_dtype_casts_and_keeps_shape():
    array = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    stored = convert_to_storage_dtype(array, "float16")
    assert stored.dtype == np.float16
    assert stored.shape == (2, 3, 4)
    assert stored.flags["C_CONTIGUOUS"]


def test_convert_to_storage_dtype_rejects_overflow():
    with pytest.raises(ValueError):
        convert_to_storage_dtype(np.array([1e6], dtype=np.float32), "float16")


def test_base64_round_trip():
    array = np.random.default_rng(0).standard_normal((4, 5, 6)).astype(np.float16)
    encoded = encode_array_to_base64(array)
    decoded = decode_base64_to_array(encoded, [4, 5, 6], "float16")
    np.testing.assert_array_equal(decoded, array)
