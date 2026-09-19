from common.config import Settings
from feature_extractor.extractor.base import FeatureExtractor
from feature_extractor.extractor.service import read_upload, extract_features_from_image


def build_extractor(settings: Settings) -> FeatureExtractor:
    backend = settings.model.backend
    if backend == "fake":
        from feature_extractor.extractor.fake import FakeExtractor
        return FakeExtractor(settings.model, settings.image.size)
    if backend == "vfm":
        from feature_extractor.extractor.vfm import VFMExtractor
        return VFMExtractor(settings.model, settings.image.size)
    raise ValueError(f"unknown model backend: {backend}")
