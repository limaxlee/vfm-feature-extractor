from pydantic import BaseModel

from .features import ModelInfo


class CheckHealthResponse(BaseModel):
    server_status: str = "healthy"


class CheckReadyResponse(BaseModel):
    server_status: str = "ready"
    backend: str
    model: ModelInfo
