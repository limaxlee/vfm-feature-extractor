# VFM Feature Extractor

FastAPI service that takes one image, resizes it to 512x512, runs the frozen DINOv3
model from the `cvpr27` codebase (`platform/feature_export/extract.py`) and returns:

- **feature vector** `patch_mean_concat`, shape `(2048,)`: per-layer patch means of layers 20 and 24, L2-normalized and concatenated
- **feature maps** `layer_{L}_feature_map`, shape `(1024, 32, 32)` per layer, on request

The model is loaded once at startup and stays resident. Nothing is written to disk per request.

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for what was built, sample responses and open checks, and [RUNNING.md](RUNNING.md) for the step-by-step guide to run it in the `vfm` container.

## Layout

```
common/             config.yaml loader (SETTINGS) and constants
feature_extractor/  main.py, routes/, schemas/, extractor/ (service + vfm/fake backends), utils/
tests/              pytest suite running against the fake backend (no GPU, no torch)
config.yaml         runtime configuration
run.sh              launch script for the container
```

## API

| Method | Path            | Description |
|--------|-----------------|-------------|
| POST   | `/features`     | multipart field `file`; query `include_feature_map=true|false` (default `false`) |
| GET    | `/health`       | process is alive |
| GET    | `/ready`        | model is loaded; returns model info |
| GET    | `/logs`         | zip of the log directory |
| GET    | `/docs`         | OpenAPI UI |

The vector is inline as floats. Each feature map, when requested, is base64 of raw
little-endian `float16` bytes in C order with `shape` and `dtype` alongside.

```python
import base64, numpy as np, requests

url = "http://SERVER:24500/features"

r = requests.post(url, files={"file": open("img.jpg", "rb")}).json()
vector = np.asarray(r["feature_vector"]["data"], dtype=np.float32)          # (2048,)

r = requests.post(url, params={"include_feature_map": "true"}, files={"file": open("img.jpg", "rb")}).json()
fmap = r["feature_maps"][1]                                                   # layer 24
arr = np.frombuffer(base64.b64decode(fmap["data"]), dtype=fmap["dtype"]).reshape(fmap["shape"])  # (1024, 32, 32)
```

## Configuration

`config.yaml` at the repo root, or `--config path` on the command line. Environment
variables override individual keys: `SERVER_PORT`, `MODEL_BACKEND`, `MODEL_ROOT`,
`MODEL_CHECKPOINT`, `MODEL_LAYERS`, `MODEL_DEVICE`, `MODEL_AMP`, `MODEL_L2_PATCHES`,
`MODEL_STORAGE_DTYPE`, `MODEL_WARMUP`, `IMAGE_SIZE`, `IMAGE_MAX_UPLOAD_MB`.

## Development (laptop)

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt   # Windows
pytest
MODEL_BACKEND=fake MODEL_WARMUP=false python -m feature_extractor.main   # API with dummy features
```

## Running in the vfm container

```bash
pip install -r requirements.txt          # torch/numpy/pillow already present, do not reinstall
bash run.sh                              # binds 0.0.0.0:24500, logs to logs/
```

`run.sh` exports the PyTorch allocator variables required by the model before torch is imported.
