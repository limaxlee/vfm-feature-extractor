from typing import Annotated
from fastapi import APIRouter, status, HTTPException, Depends, File, Query, UploadFile

from common.config import SETTINGS
from common.constants import SUPPORTED_CONTENT_TYPES
from feature_extractor.dependencies import get_extractor
from feature_extractor.extractor import FeatureExtractor, read_upload, extract_features_from_image
from feature_extractor.schemas import ExtractFeaturesQuery, ExtractFeaturesResponse
from feature_extractor.utils import prepare_image

router = APIRouter(tags=["features"])


@router.post("/features", response_model=ExtractFeaturesResponse, status_code=status.HTTP_200_OK)
async def extract_features(
    query: Annotated[ExtractFeaturesQuery, Query()],
    file: UploadFile = File(..., description="Image file (jpeg, png, bmp, webp, tiff)"),
    extractor: FeatureExtractor = Depends(get_extractor),
):
    if file.content_type and file.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type {file.content_type}"
        )

    try:
        data = await read_upload(file)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(e))

    try:
        prepared = prepare_image(data, SETTINGS.image.size)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    try:
        return await extract_features_from_image(prepared, file.filename, file.content_type, query, extractor)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
