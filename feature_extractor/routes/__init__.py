from fastapi import APIRouter

from .health import router as health_router
from .logs import router as logs_router
from .features import router as features_router

router = APIRouter()
router.include_router(health_router)
router.include_router(logs_router)
router.include_router(features_router)
