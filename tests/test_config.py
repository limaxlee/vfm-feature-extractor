import pytest

from common.config import Settings, load_config, _set_nested_config


def test_set_nested_config_creates_groups():
    config = {}
    _set_nested_config(config, "model.device", "cpu")
    _set_nested_config(config, "server_port", 1)
    assert config == {"model": {"device": "cpu"}, "server_port": 1}


def test_set_nested_config_rejects_non_mapping():
    with pytest.raises(ValueError):
        _set_nested_config({"model": "oops"}, "model.device", "cpu")


def test_load_config_reads_yaml_and_env_overrides(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("server_port: 1000\nmodel:\n  device: cuda:0\n  layers: [20, 24]\n", encoding="utf-8")

    config = load_config(str(config_file), environ={
        "SERVER_PORT": "24500",
        "MODEL_LAYERS": "24",
        "MODEL_WARMUP": "false",
        "MODEL_L2_PATCHES": "yes",
    })

    assert config["server_port"] == 24500
    assert config["model"]["device"] == "cuda:0"
    assert config["model"]["layers"] == [24]
    assert config["model"]["warmup"] is False
    assert config["model"]["l2_patches"] is True


def test_load_config_invalid_env_value(tmp_path):
    with pytest.raises(ValueError):
        load_config(str(tmp_path / "missing.yaml"), environ={"SERVER_PORT": "abc"})


def test_settings_defaults(tmp_path):
    settings = Settings(**load_config(str(tmp_path / "missing.yaml"), environ={}))
    assert settings.server_port == 24500
    assert settings.model.backend == "vfm"
    assert settings.model.root == "/work/cvpr27"
    assert settings.model.checkpoint == "export_snap3"
    assert settings.model.layers == [20, 24]
    assert settings.model.storage_dtype == "float16"
    assert settings.image.size == 512
    assert settings.image.max_upload_mb == 20


def test_settings_rejects_bad_values(tmp_path):
    with pytest.raises(ValueError):
        Settings(**load_config(str(tmp_path / "missing.yaml"), environ={"MODEL_BACKEND": "other"}))
    with pytest.raises(ValueError):
        Settings(**load_config(str(tmp_path / "missing.yaml"), environ={"MODEL_STORAGE_DTYPE": "bf16"}))
