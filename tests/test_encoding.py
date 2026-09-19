import json
from io import BytesIO

import numpy as np
import pytest

from common.constants import ExtractorKeys
from feature_extractor.utils import (
    convert_to_storage_dtype, encode_array_to_base64, decode_base64_to_array, pack_arrays_to_npz
)


def test_to_storage_dtype_casts_and_keeps_shape():
    array = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    stored = convert_to_storage_dtype(array, "float16")
    assert stored.dtype == np.float16
    assert stored.shape == (2, 3, 4)
    assert stored.flags["C_CONTIGUOUS"]


def test_to_storage_dtype_rejects_overflow():
    with pytest.raises(ValueError):
        convert_to_storage_dtype(np.array([1e6], dtype=np.float32), "float16")


def test_base64_round_trip():
    array = np.random.default_rng(0).standard_normal((4, 5, 6)).astype(np.float16)
    encoded = encode_array_to_base64(array)
    decoded = decode_base64_to_array(encoded, [4, 5, 6], "float16")
    np.testing.assert_array_equal(decoded, array)


def test_npz_round_trip_matches_extract_layout():
    arrays = {
        "patch_mean_concat": np.ones(2048, dtype=np.float16),
        "layer_24_feature_map": np.zeros((1024, 32, 32), dtype=np.float16),
    }
    data = pack_arrays_to_npz(arrays, {"source": "x.jpg"})

    with np.load(BytesIO(data), allow_pickle=False) as loaded:
        assert set(loaded.files) == {"patch_mean_concat", "layer_24_feature_map", ExtractorKeys.METADATA}
        assert loaded["patch_mean_concat"].shape == (2048,)
        assert loaded["patch_mean_concat"].dtype == np.float16
        assert loaded["layer_24_feature_map"].shape == (1024, 32, 32)
        metadata = json.loads(str(loaded[ExtractorKeys.METADATA]))

    assert metadata["source"] == "x.jpg"
    assert metadata["shapes"] == {"patch_mean_concat": [2048], "layer_24_feature_map": [1024, 32, 32]}
