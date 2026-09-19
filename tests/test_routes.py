import json
from io import BytesIO

import numpy as np

from common.constants import HIDDEN_SIZE, ExtractorKeys
from feature_extractor.utils import decode_base64_to_array
from tests.conftest import make_image_bytes


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"server_status": "healthy"}


def test_ready_reports_model_info(client, settings):
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["server_status"] == "ready"
    assert body["backend"] == "fake"
    assert body["model"]["hidden_size"] == HIDDEN_SIZE
    assert body["model"]["layers"] == settings.model.layers


def test_extract_features_json(client, settings, jpeg_bytes):
    response = client.post("/features", files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")})
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["image"]["filename"] == "photo.jpg"
    assert body["image"]["original_hw"] == [480, 640]
    assert body["image"]["input_hw"] == [512, 512]
    assert body["image"]["grid_hw"] == [32, 32]
    assert body["storage_dtype"] == settings.model.storage_dtype

    vector = body["feature_vector"]
    assert vector["name"] == "patch_mean_concat"
    assert vector["shape"] == [HIDDEN_SIZE * len(settings.model.layers)]
    assert vector["encoding"] == "list"
    assert len(vector["data"]) == HIDDEN_SIZE * len(settings.model.layers)
    assert abs(np.linalg.norm(vector["data"]) - 1.0) < 1e-2

    maps = body["feature_maps"]
    assert [m["layer"] for m in maps] == settings.model.layers
    for payload in maps:
        assert payload["name"] == f"layer_{payload['layer']}_feature_map"
        assert payload["shape"] == [HIDDEN_SIZE, 32, 32]
        assert payload["encoding"] == "base64"
        decoded = decode_base64_to_array(payload["data"], payload["shape"], payload["dtype"])
        assert decoded.shape == (HIDDEN_SIZE, 32, 32)
        assert np.isfinite(decoded).all()


def test_extract_features_is_deterministic(client, jpeg_bytes):
    first = client.post("/features", files={"file": ("a.jpg", jpeg_bytes, "image/jpeg")}).json()
    second = client.post("/features", files={"file": ("b.jpg", jpeg_bytes, "image/jpeg")}).json()
    assert first["feature_vector"]["data"] == second["feature_vector"]["data"]


def test_extract_features_without_feature_map(client, jpeg_bytes):
    response = client.post(
        "/features",
        params={"include_feature_map": "false"},
        files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["feature_maps"] == []


def test_extract_features_npz(client, settings, png_rgba_bytes):
    response = client.post(
        "/features",
        params={"format": "npz"},
        files={"file": ("shot.png", png_rgba_bytes, "image/png")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert "shot.npz" in response.headers["content-disposition"]

    with np.load(BytesIO(response.content), allow_pickle=False) as loaded:
        expected = {f"layer_{layer}_feature_map" for layer in settings.model.layers}
        expected |= {"patch_mean_concat", ExtractorKeys.METADATA}
        assert set(loaded.files) == expected
        assert loaded["patch_mean_concat"].shape == (HIDDEN_SIZE * len(settings.model.layers),)
        assert loaded["patch_mean_concat"].dtype == np.dtype(settings.model.storage_dtype)
        metadata = json.loads(str(loaded[ExtractorKeys.METADATA]))

    assert metadata["source"] == "shot.png"
    assert metadata["original_hw"] == [700, 300]
    assert metadata["scale_yx"] == [512 / 700, 512 / 300]
    assert metadata["model"]["layers"] == settings.model.layers


def test_extract_features_rejects_unsupported_content_type(client):
    response = client.post("/features", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_extract_features_rejects_corrupt_image(client):
    response = client.post("/features", files={"file": ("bad.jpg", b"not an image", "image/jpeg")})
    assert response.status_code == 400


def test_extract_features_rejects_oversized_upload(client, settings):
    payload = b"\xff" * (settings.image.max_upload_mb * 1024 * 1024 + 1)
    response = client.post("/features", files={"file": ("big.jpg", payload, "image/jpeg")})
    assert response.status_code == 413


def test_extract_features_accepts_octet_stream(client):
    data = make_image_bytes(size=(100, 100), fmt="PNG")
    response = client.post("/features", files={"file": ("blob", data, "application/octet-stream")})
    assert response.status_code == 200
