from io import BytesIO
from PIL import Image, ImageOps, UnidentifiedImageError

from common.constants import SUPPORTED_IMAGE_MODES
from feature_extractor.schemas import PreparedImage


def decode_image(data: bytes) -> Image.Image:
    if not data:
        raise ValueError("empty upload")

    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            decoded = image.copy()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError(f"cannot decode image: {error}") from error

    if decoded.mode not in SUPPORTED_IMAGE_MODES:
        raise ValueError(
            f"unsupported image mode {decoded.mode}; convert high-bit-depth data to 8-bit first"
        )

    return decoded


def prepare_image(data: bytes, size: int) -> PreparedImage:
    """Decode, apply EXIF orientation, convert to RGB and resize to size x size.

    Mirrors preprocess_image in extract.py (bilinear, no aspect preservation,
    no crop) so the model's own resize step becomes a no-op.
    """
    decoded = decode_image(data)
    stored_hw = [decoded.height, decoded.width]
    orientation = int(decoded.getexif().get(274, 1))

    image = ImageOps.exif_transpose(decoded).convert("RGB")
    original_hw = [image.height, image.width]

    if (image.height, image.width) != (size, size):
        image = image.resize((size, size), resample=Image.Resampling.BILINEAR)

    # Orientation is applied. Drop the EXIF so the model's own exif_transpose cannot rotate twice.
    image.info.pop("exif", None)

    return PreparedImage(
        image=image,
        original_hw=original_hw,
        stored_hw=stored_hw,
        exif_orientation=orientation
    )
