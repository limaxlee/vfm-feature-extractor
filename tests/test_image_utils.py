from io import BytesIO

import pytest
from PIL import Image

from feature_extractor.utils import decode_image, prepare_image
from tests.conftest import make_image_bytes


def test_prepare_image_resizes_to_square(jpeg_bytes):
    prepared = prepare_image(jpeg_bytes, 512)
    assert prepared.image.mode == "RGB"
    assert prepared.image.size == (512, 512)
    assert prepared.original_hw == [480, 640]
    assert prepared.stored_hw == [480, 640]
    assert prepared.exif_orientation == 1


def test_prepare_image_converts_rgba_to_rgb(png_rgba_bytes):
    prepared = prepare_image(png_rgba_bytes, 512)
    assert prepared.image.mode == "RGB"
    assert prepared.image.size == (512, 512)
    assert prepared.original_hw == [700, 300]


def test_prepare_image_applies_exif_orientation():
    image = Image.new("RGB", (640, 480), (1, 2, 3))
    exif = image.getexif()
    exif[274] = 6  # rotate 90 degrees clockwise on display
    with BytesIO() as buffer:
        image.save(buffer, format="JPEG", exif=exif.tobytes())
        data = buffer.getvalue()

    prepared = prepare_image(data, 512)
    assert prepared.exif_orientation == 6
    assert prepared.stored_hw == [480, 640]
    assert prepared.original_hw == [640, 480]
    assert prepared.image.size == (512, 512)
    # the model applies exif_transpose again; nothing must be left for it to act on
    assert "exif" not in prepared.image.info
    assert prepared.image.getexif().get(274, 1) == 1


def test_prepare_image_keeps_already_square_input():
    data = make_image_bytes(size=(512, 512))
    prepared = prepare_image(data, 512)
    assert prepared.image.size == (512, 512)


def test_decode_image_rejects_corrupt_bytes():
    with pytest.raises(ValueError):
        decode_image(b"definitely not an image")


def test_decode_image_rejects_empty():
    with pytest.raises(ValueError):
        decode_image(b"")


def test_decode_image_rejects_high_bit_depth():
    data = make_image_bytes(size=(32, 32), mode="I;16", fmt="PNG")
    with pytest.raises(ValueError):
        decode_image(data)
