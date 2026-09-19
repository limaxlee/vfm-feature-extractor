import time
import logging
from fastapi import UploadFile
from fastapi.responses import Response
from fastapi.concurrency import run_in_threadpool

from common.config import SETTINGS
from common.constants import UPLOAD_CHUNK_BYTES, ExtractorKeys, ResponseFormat
from feature_extractor.extractor.base import FeatureExtractor
from feature_extractor.schemas import (
    PreparedImage, ExtractedFeatures, ExtractFeaturesQuery, ArrayPayload, FeatureMapPayload, ImageInfo, ModelInfo,
    ExtractFeaturesResponse
)
from feature_extractor.utils import convert_to_storage_dtype, encode_array_to_base64, pack_arrays_to_npz

logger = logging.getLogger(__name__)


async def read_upload(file: UploadFile) -> bytes:
    """Read the upload in chunks; raise ValueError as soon as it exceeds the configured limit."""
    max_bytes = SETTINGS.image.max_upload_mb * 1024 * 1024
    chunks, total = [], 0
    while chunk := await file.read(UPLOAD_CHUNK_BYTES):
        total += len(chunk)
        if total > max_bytes:
            logger.warning(f"Rejected upload {file.filename}: exceeds {SETTINGS.image.max_upload_mb} MB")
            raise ValueError(f"Upload exceeds {SETTINGS.image.max_upload_mb} MB")
        chunks.append(chunk)

    return b"".join(chunks)


async def extract_features_from_image(
    prepared: PreparedImage, filename: str | None, content_type: str | None,
    query: ExtractFeaturesQuery, extractor: FeatureExtractor
) -> ExtractFeaturesResponse | Response:
    """Run the model on an already prepared image and build the response in the requested format."""
    try:
        started = time.perf_counter()
        result = await run_in_threadpool(extractor.extract, prepared.image)
        elapsed = (time.perf_counter() - started) * 1000

        if query.format is ResponseFormat.NPZ:
            response = _build_npz_response(filename, prepared, result, extractor.info, query.include_feature_map)
        else:
            response = _build_feature_response(
                filename, content_type, prepared, result, extractor.info, query.include_feature_map
            )

        logger.info(
            f"Extracted features from {filename} ({prepared.original_hw[1]}x{prepared.original_hw[0]}): "
            f"vector {tuple(result.feature_vector.shape)}, {len(result.feature_maps)} feature maps, "
            f"format {query.format}, inference {elapsed:.1f} ms"
        )
        return response
    except Exception as e:
        logger.exception(f"Failed to extract features from {filename}: {str(e)}")
        raise


def _build_image_info(filename: str | None, content_type: str | None, prepared: PreparedImage,
                      result: ExtractedFeatures) -> ImageInfo:
    size = SETTINGS.image.size
    return ImageInfo(
        filename=filename,
        content_type=content_type,
        stored_hw=prepared.stored_hw,
        original_hw=prepared.original_hw,
        exif_orientation=prepared.exif_orientation,
        input_hw=result.image_meta.get("input_hw", [size, size]),
        grid_hw=result.image_meta.get("grid_hw", []),
    )


def _build_feature_response(
    filename: str | None, content_type: str | None, prepared: PreparedImage,
    result: ExtractedFeatures, model_info: dict, include_feature_map: bool
) -> ExtractFeaturesResponse:
    dtype = SETTINGS.model.storage_dtype
    vector = convert_to_storage_dtype(result.feature_vector, dtype)

    feature_maps = []
    if include_feature_map:
        for layer, array in result.feature_maps.items():
            stored = convert_to_storage_dtype(array, dtype)
            feature_maps.append(FeatureMapPayload(
                name=f"layer_{layer}_{ExtractorKeys.FEATURE_MAP}", layer=layer, shape=list(stored.shape),
                dtype=dtype, encoding="base64", data=encode_array_to_base64(stored)
            ))

    return ExtractFeaturesResponse(
        image=_build_image_info(filename, content_type, prepared, result),
        model=ModelInfo(**model_info),
        storage_dtype=dtype,
        feature_vector=ArrayPayload(
            name=ExtractorKeys.FEATURE_VECTOR, shape=list(vector.shape), dtype=dtype,
            encoding="list", data=vector.astype("float32").tolist()
        ),
        feature_maps=feature_maps,
    )


def _build_npz_response(filename: str | None, prepared: PreparedImage, result: ExtractedFeatures,
                        model_info: dict, include_feature_map: bool) -> Response:
    dtype = SETTINGS.model.storage_dtype
    arrays = {ExtractorKeys.FEATURE_VECTOR: convert_to_storage_dtype(result.feature_vector, dtype)}
    if include_feature_map:
        for layer, array in result.feature_maps.items():
            arrays[f"layer_{layer}_{ExtractorKeys.FEATURE_MAP}"] = convert_to_storage_dtype(array, dtype)

    # The model saw the already-resized image, so its stored/original/scale fields describe
    # the 512x512 input. Overwrite them with the values of the real upload.
    size = SETTINGS.image.size
    metadata = {
        "schema_version": 1,
        "source": filename,
        **result.image_meta,
        "stored_hw": prepared.stored_hw,
        "original_hw": prepared.original_hw,
        "exif_orientation": prepared.exif_orientation,
        "scale_yx": [size / prepared.original_hw[0], size / prepared.original_hw[1]],
        "model": model_info,
        "storage_dtype": dtype,
    }
    stem = (filename or "features").rsplit(".", 1)[0] or "features"
    return Response(
        content=pack_arrays_to_npz(arrays, metadata),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{stem}.npz"'}
    )
