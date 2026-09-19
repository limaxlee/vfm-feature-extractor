# VFM Feature Extractor — Implementation Notes

A FastAPI service that takes one image, resizes it to 512×512, runs the frozen DINOv3
model from the `cvpr27` codebase, and returns the image's **feature vector** and
**per-layer feature maps**. The model is loaded once at startup and stays resident.
Nothing is written to disk per request.

The model code is not copied into this repository. The service imports
`platform/feature_export/extract.py` from the `cvpr27` checkout inside the `vfm`
container and drives its `DinoFeatureExtractor` class directly.

---

## 1. What was implemented

| Area | Summary |
|------|---------|
| `POST /features` | Multipart image upload → feature vector `(2048,)`, plus feature maps `(1024, 32, 32)` per layer when `include_feature_map=true`. JSON output. |
| `GET /health`, `GET /ready` | Liveness and readiness. `/ready` returns 503 until the model is loaded, then the model description. |
| `GET /logs` | Zip of the log directory, timestamped filename. |
| Model backends | `vfm` (real DINOv3, GPU, inside the container) and `fake` (deterministic arrays with the real shapes, no torch). Selected in `config.yaml`. |
| Config | `config.yaml` loaded into `SETTINGS` (pydantic); every key overridable by environment variable. |
| Logging | Rotating file + console via `utils/logger.py`; request/response lines from the middleware; per-extraction line from the service. |
| Tests | 29 tests against the fake backend. Run on a laptop, no GPU, no torch. |

### Project layout

```
common/
├── config.py                 config.yaml → SETTINGS (pydantic BaseSettings); env overrides via _ENV_MAP
└── constants.py              ROOT_DIR, IMAGE_SIZE=512, layer/patch/hidden defaults, ExtractorKeys,
                              supported content types / image modes, torch allocator env vars
feature_extractor/
├── main.py                   FastAPI app, lifespan (build extractor once + warmup), CORS, middleware, router
├── dependencies.py           get_extractor() for Depends()
├── middleware.py             log_requests_middleware
├── routes/                   one file per concern; __init__ aggregates into one router
│   ├── health.py             check_health, check_ready
│   ├── features.py           extract_features
│   └── logs.py               download_logs
├── schemas/
│   ├── health.py             CheckHealthResponse, CheckReadyResponse
│   └── features.py           PreparedImage, ExtractedFeatures, ExtractFeaturesRequest, ExtractFeaturesResponse,
│                             ArrayPayload, FeatureMapPayload, ImageInfo, ModelInfo
├── extractor/
│   ├── base.py               FeatureExtractor protocol
│   ├── service.py            read_upload, extract_features_from_image (+ private response builders)
│   ├── vfm.py                real backend wrapping DinoFeatureExtractor
│   └── fake.py               test backend
└── utils/
    ├── logger.py             initialize_logger, LogConfig, get_logs_zip_file
    ├── image.py              decode_image, prepare_image (EXIF transpose → RGB → 512×512 bilinear)
    └── encoding.py           convert_to_storage_dtype, encode_array_to_base64, decode_base64_to_array
tests/                        pytest suite (fake backend)
docs/samples/                 real responses captured from the app (see §4)
config.yaml                   runtime configuration
run.sh                        launch script for the container
```

### Request flow

```
upload ──► routes/features.py
            415 if content type is not an image type
            read_upload()      → 413 if larger than image.max_upload_mb
            prepare_image()    → 400 if not a decodable 8-bit image
            extract_features_from_image()
                extractor.extract()   (threadpool + lock, one forward pass)
                build the response    → 500 on any unexpected failure
```

---

## 2. Configuration

`config.yaml` at the repo root, or `--config <path>` on the command line.

```yaml
server_port: 24500

model:
  backend: vfm            # vfm | fake
  root: /work/cvpr27      # <root>/platform is added to sys.path
  checkpoint: export_snap3   # relative paths resolve against root
  layers: [20, 24]
  device: cuda:0
  amp: bf16               # bf16 | fp32
  l2_patches: false
  storage_dtype: float16  # dtype of arrays in the response: float16 | float32
  warmup: true            # one dummy forward at startup

image:
  size: 512               # every input is resized to size × size
  max_upload_mb: 20
```

Environment overrides: `SERVER_PORT`, `MODEL_BACKEND`, `MODEL_ROOT`, `MODEL_CHECKPOINT`,
`MODEL_LAYERS` (e.g. `"20 24"`), `MODEL_DEVICE`, `MODEL_AMP`, `MODEL_L2_PATCHES`,
`MODEL_STORAGE_DTYPE`, `MODEL_WARMUP`, `IMAGE_SIZE`, `IMAGE_MAX_UPLOAD_MB`.

---

## 3. API

| Method | Path | Query | Returns |
|--------|------|-------|---------|
| POST | `/features` | `include_feature_map=true\|false` (default `false`) | `ExtractFeaturesResponse` |
| GET | `/health` | | `{"server_status": "healthy"}` |
| GET | `/ready` | | model description, or 503 while loading |
| GET | `/logs` | | `application/zip` |
| GET | `/docs` | | OpenAPI UI |

Multipart field name: `file`. Accepted content types: `image/jpeg`, `image/png`, `image/bmp`,
`image/webp`, `image/tiff`, `application/octet-stream` (validated by decoding), or none.

### Error codes

| Code | When |
|------|------|
| 400 | Upload is empty, not decodable, or a high-bit-depth mode (`I;16`, `F`, …) |
| 413 | Upload larger than `image.max_upload_mb` |
| 415 | Content type present but not an image type |
| 503 | Model not loaded yet (`/features`, `/ready`) |
| 500 | Model or serialization failure; see logs |

Error body: `{"detail": "<message>"}`.

---

## 4. Sample request and response

### curl

```bash
# Default: feature vector only (~30 KB)
curl -F "file=@photo.jpg" "http://SERVER:24500/features" -o response.json

# Vector plus both feature maps (~5.6 MB)
curl -F "file=@photo.jpg" "http://SERVER:24500/features?include_feature_map=true" -o response.json
```

### Python

```python
import base64
import numpy as np
import requests

url = "http://SERVER:24500/features"

# Vector only
r = requests.post(url, files={"file": open("photo.jpg", "rb")}).json()
vector = np.asarray(r["feature_vector"]["data"], dtype=np.float32)          # (2048,)

# Vector plus feature maps
r = requests.post(url, params={"include_feature_map": "true"}, files={"file": open("photo.jpg", "rb")}).json()
fmap = r["feature_maps"][1]                                                   # layer 24
arr = np.frombuffer(base64.b64decode(fmap["data"]), dtype=fmap["dtype"]).reshape(fmap["shape"])  # (1024, 32, 32)
```

### JSON response (trimmed)

Files captured from the running app are in `docs/samples/`. Long arrays are truncated, and each
file says so in its `_note` field:

- `extract_features_response.json` — default request, vector only
- `extract_features_response_with_maps.json` — `include_feature_map=true`
- `check_ready_response.json` — `GET /ready`
- `error_response.json` — a 400

They were generated with the **fake** backend, so `model.model_type` is `fake`, `device` is `cpu`
and `effective_amp` is `fp32`. With the real backend those read `dinov3_vit`, `cuda:0`, `bf16`.
Shapes, keys and sizes are identical.

```json
{
  "image": {
    "filename": "photo.jpg",
    "content_type": "image/jpeg",
    "stored_hw": [480, 640],
    "original_hw": [480, 640],
    "exif_orientation": 1,
    "input_hw": [512, 512],
    "grid_hw": [32, 32]
  },
  "model": {
    "checkpoint": "/work/cvpr27/export_snap3",
    "model_type": "dinov3_vit",
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "patch_size": [16, 16],
    "num_register_tokens": 4,
    "layers": [20, 24],
    "device": "cuda:0",
    "effective_amp": "bf16"
  },
  "storage_dtype": "float16",
  "feature_vector": {
    "name": "patch_mean_concat",
    "shape": [2048],
    "dtype": "float16",
    "encoding": "list",
    "data": [-0.0133, -0.0157, 0.0070, 0.0513, "... 2048 floats"]
  },
  "feature_maps": [
    {
      "name": "layer_20_feature_map",
      "layer": 20,
      "shape": [1024, 32, 32],
      "dtype": "float16",
      "encoding": "base64",
      "data": "W77JuBy9HTtYu60v7CwbuuI82zvDtqG/..."
    },
    {
      "name": "layer_24_feature_map",
      "layer": 24,
      "shape": [1024, 32, 32],
      "dtype": "float16",
      "encoding": "base64",
      "data": "/b7NO5yzRzlNwlu2yjnhtJo5mDtCPTu6..."
    }
  ]
}
```

Field notes:

- `stored_hw` — size as stored in the file. `original_hw` — after EXIF orientation is applied. `input_hw` — what the model saw (always 512×512). `grid_hw` — patch grid, `input / patch_size`.
- `feature_vector.data` is a plain JSON list. `feature_maps[].data` is base64 of the raw little-endian `float16` bytes in C order; decode with `np.frombuffer(...).reshape(shape)`.
- By default `feature_maps` is `[]`. The JSON above was requested with `include_feature_map=true`.

### Sizes and cost (one image)

| Request | Size | Server-side time beyond the model forward |
|---------|------|-------------------------------------------|
| default, vector only | ~30 KB | ~1 ms |
| `include_feature_map=true`, both maps | 5.6 MB | ~35 ms (float16 cast + base64 of two 2 MB maps) |

Each feature map is 2 MB of float16; base64 inflates it by a third. On a 1 Gbps LAN the 5.6 MB
transfer takes about 45 ms; on a 100 Mbps link about 450 ms. Request feature maps only when
they will be used.

---

## 5. How extraction maps to `extract.py`

`extractor/vfm.py` does what the reference command does, minus the files:

```bash
python platform/feature_export/extract.py --input img.jpg --out DIR \
  --checkpoint export_snap3 --layers 20 24 --features patch_mean_concat \
  --input-size 512 --batch-size 1 --device cuda:0 --storage-dtype float16
```

| Step | CLI | API |
|------|-----|-----|
| Import | run as a script | `sys.path.insert(0, "<root>/platform")`, `from feature_export.extract import DinoFeatureExtractor` (the path the file's own docstring documents) |
| Checkpoint | `export_snap3` relative to cwd `cvpr27` | `os.path.join(model.root, model.checkpoint)` → `/work/cvpr27/export_snap3` |
| Constructor | `layers=(20, 24)`, `device`, `amp`, `l2_patches` from CLI + YAML | same arguments from `config.yaml` |
| Preprocess | `preprocess_image`: EXIF transpose → RGB → bilinear resize to 512×512 → /255 → ImageNet mean/std | `utils/image.py` does transpose → RGB → bilinear 512×512 and strips EXIF, then passes `input_size=(512, 512)`; the model's own resize is a no-op and normalization happens once, in `preprocess_image` |
| Forward | `extract_images(..., batch_size=1, feature_names=["patch_mean_concat"])` | `extract_images([image], input_size=(512, 512), resize_mode="resize", batch_size=1, feature_names=("feature_map", "patch_mean_concat"))` |
| Output | writes `patch_mean_concat` to `.npz` | takes `patch_mean_concat` and `layer_{L}_feature_map` from the record, keeps CPU float32, casts to `storage_dtype` at serialization with the same finite check as `write_record` |
| Concurrency | one process, one image | one resident model, `threading.Lock` around the forward, inference in a threadpool |

Definitions, from `extract_batch`:

- **feature map** (`layer_{L}_feature_map`): patch tokens of block *L* after the model's final
  LayerNorm, reshaped to `(hidden, grid_h, grid_w)`. Not L2-normalized unless `l2_patches` is true.
- **feature vector** (`patch_mean_concat`): for each layer, mean over patch tokens after LayerNorm
  then L2-normalize; concatenate the layers; L2-normalize again. Shape `(1024 × len(layers),)`.

Both come from a single forward pass. The vector returned by the API is the same tensor the
CLI writes to its `.npz` for the same image, cast to the same dtype.

---

## 6. Things only the container can answer

These could not be verified on a laptop and need one check each on the server.

1. **`l2_patches` and `resize_mode` come from the model's YAML.** The reference command does not
   pass them, so `extract.py` reads them from
   `platform/feature_export/configs/manufacturing_snap3.yaml`. The API assumes `l2_patches: false`
   and `resize_mode: resize`. `l2_patches` does not change the vector, but if the YAML sets it to
   `true` the feature maps would be per-patch normalized and `config.yaml` must match.

   ```bash
   cat /work/cvpr27/platform/feature_export/configs/manufacturing_snap3.yaml
   ```

2. **float16 range of the feature maps.** The CLI has only ever exported `patch_mean_concat`
   (unit norm, safe in float16). Feature maps are LayerNorm outputs, and ViT models sometimes have
   a few channels with very large activations. If a value exceeds the float16 range the cast raises
   and the request returns 500. If that happens on real images, set `storage_dtype: float32` in
   `config.yaml`. The first real request will show it.

3. **Python ≥ 3.11 and pydantic 2 inside the container.** `StrEnum` needs 3.11; the schemas need
   pydantic 2. Check before installing:

   ```bash
   python --version
   pip list 2>/dev/null | grep -i -E "pydantic|fastapi|uvicorn"
   ```

   If pydantic 1 is pinned by another package, stop and discuss before upgrading.

4. **First real image.** Run one request with `include_feature_map=true` for the same image the
   CLI exported to `sample001/00001.npz`, and compare `feature_vector.data` to the CLI's
   `patch_mean_concat`. They should match to float16 precision. RUNNING.md §9 has the script.

---

## 7. Running

### Laptop (fake backend)

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt   # Windows
pytest
MODEL_BACKEND=fake MODEL_WARMUP=false python -m feature_extractor.main
curl -F "file=@some.jpg" "http://localhost:24500/features?include_feature_map=false"
```

### Container (`vfm`)

Deployment steps are agreed separately. In short: the code lives inside the container, only the
web layer is pip-installed (torch, numpy, pillow, transformers are already there and must not be
reinstalled), and `run.sh` starts the service on port 24500, which the container already publishes.

```bash
cd /work/vfm-feature-extractor
pip install -r requirements.txt
bash run.sh                       # exports torch allocator vars, logs to logs/
curl http://localhost:24500/ready
```

Start in the background from the host with:

```bash
sudo docker exec -d vfm bash /work/vfm-feature-extractor/run.sh
```

---

## 8. Design decisions

- **Model loads once.** A subprocess per request would reload weights and initialize CUDA every time and force disk writes. The lifespan builds the extractor, runs one warmup forward, and keeps it on `app.state`.
- **Resize in the API, not only in the model.** The requirement is 512×512 for every input. `utils/image.py` guarantees it, and the model receives an image its own transform leaves untouched.
- **Both outputs from one pass.** `feature_names` selects the two keys; the script only moves requested tensors to CPU.
- **Feature maps are opt-in.** They are 5.6 MB per response, so the default returns only the 30 KB vector and `include_feature_map=true` adds the maps.
- **Single worker, serialized GPU.** One GPU, one model, batch size 1. Inference runs in a threadpool so the event loop stays responsive; a lock queues concurrent requests.
- **Fake backend for tests.** The whole HTTP surface, image handling and encoding are tested without torch.
- **Errors are built-ins.** Utilities raise `ValueError`; routes map them to status codes. Deliberate 4xx raises sit outside `try` blocks so they are never rewrapped as 500.
