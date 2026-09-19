import os
import yaml
import argparse
from typing import Any, Literal
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from common.constants import ROOT_DIR, DEFAULT_LAYERS, IMAGE_SIZE


def _to_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


def _to_int_list(value: str) -> list[int]:
    return [int(part) for part in value.replace(",", " ").split()]


_ENV_MAP = {
    "SERVER_PORT": ("server_port", int),
    "MODEL_BACKEND": ("model.backend", str),
    "MODEL_ROOT": ("model.root", str),
    "MODEL_CHECKPOINT": ("model.checkpoint", str),
    "MODEL_LAYERS": ("model.layers", _to_int_list),
    "MODEL_DEVICE": ("model.device", str),
    "MODEL_AMP": ("model.amp", str),
    "MODEL_L2_PATCHES": ("model.l2_patches", _to_bool),
    "MODEL_STORAGE_DTYPE": ("model.storage_dtype", str),
    "MODEL_WARMUP": ("model.warmup", _to_bool),
    "IMAGE_SIZE": ("image.size", int),
    "IMAGE_MAX_UPLOAD_MB": ("image.max_upload_mb", int),
}


def _set_nested_config(config: dict[str, Any], key: str, value: Any) -> None:
    *groups, values = key.split(".")
    node = config
    for group in groups:
        node = node.setdefault(group, {})
        if not isinstance(node, dict):
            raise ValueError(f"Config key {group} is invalid: {type(node).__name__}")

    node[values] = value


def load_config(config_path: str | None = None, environ: dict[str, str] | None = None) -> dict[str, Any]:
    if config_path is None:
        default_config = os.path.join(ROOT_DIR, "config.yaml")
        parser = argparse.ArgumentParser(description="VFM Feature Extractor Configurations")
        parser.add_argument("--config", "-c", type=str, help="Path to the config.yaml", default=default_config)
        args, _ = parser.parse_known_args()
        config_path = args.config

    environ = os.environ if environ is None else environ

    config = {}
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            file_config = yaml.safe_load(f) or {}

        if not isinstance(file_config, dict):
            raise ValueError(f"Config file is invalid: {config_path}")

        config.update(file_config)

    for env_name, (key, caster) in _ENV_MAP.items():
        env_value = environ.get(env_name)
        if env_value is None:
            continue

        try:
            value = caster(env_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid value for {env_name}: {env_value!r}") from error

        _set_nested_config(config, key, value)

    return config


class ModelConfig(BaseModel):
    backend: Literal["vfm", "fake"] = "vfm"
    root: str = "/work/cvpr27"
    checkpoint: str = "export_snap3"
    layers: list[int] = list(DEFAULT_LAYERS)
    device: str = "cuda:0"
    amp: Literal["bf16", "fp32"] = "bf16"
    l2_patches: bool = False
    storage_dtype: Literal["float16", "float32"] = "float16"
    warmup: bool = True


class ImageConfig(BaseModel):
    size: int = IMAGE_SIZE
    max_upload_mb: int = 20


class Settings(BaseSettings, extra="allow"):
    server_port: int = 24500
    model: ModelConfig = ModelConfig()
    image: ImageConfig = ImageConfig()


SETTINGS = Settings(**load_config())
