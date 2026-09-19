from fastapi import APIRouter, status, HTTPException, Request

from common.config import SETTINGS
from feature_extractor.schemas import CheckHealthResponse, CheckReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=CheckHealthResponse, status_code=status.HTTP_200_OK)
async def check_health():
    return CheckHealthResponse()


@router.get("/ready", response_model=CheckReadyResponse, status_code=status.HTTP_200_OK)
async def check_ready(request: Request):
    extractor = getattr(request.app.state, "extractor", None)
    if extractor is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded")

    try:
        return CheckReadyResponse(backend=SETTINGS.model.backend, model=extractor.info)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
