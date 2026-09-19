from fastapi import Request, HTTPException, status

from feature_extractor.extractor import FeatureExtractor


def get_extractor(request: Request) -> FeatureExtractor:
    extractor = getattr(request.app.state, "extractor", None)
    if extractor is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="model is not loaded")
    return extractor
