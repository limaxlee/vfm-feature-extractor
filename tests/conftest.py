import os
from io import BytesIO

import pytest
from PIL import Image

# SETTINGS is a module-level singleton, so the test backend must be selected
# before anything under feature_extractor is imported.
os.environ.setdefault("MODEL_BACKEND", "fake")
os.environ.setdefault("MODEL_WARMUP", "false")
os.environ.setdefault("IMAGE_MAX_UPLOAD_MB", "1")


def make_image_bytes(size=(640, 480), mode="RGB", fmt="JPEG", color=(120, 60, 200)) -> bytes:
    image = Image.new(mode, size, color if mode in {"RGB", "RGBA"} else 128)
    with BytesIO() as buffer:
        image.save(buffer, format=fmt)
        return buffer.getvalue()


@pytest.fixture(scope="session")
def settings():
    from common.config import SETTINGS
    return SETTINGS


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from feature_extractor.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def jpeg_bytes() -> bytes:
    return make_image_bytes()


@pytest.fixture
def png_rgba_bytes() -> bytes:
    return make_image_bytes(size=(300, 700), mode="RGBA", fmt="PNG", color=(10, 20, 30, 255))
