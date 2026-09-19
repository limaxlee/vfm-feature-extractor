# Storing the feature vector in Milvus for image similarity search

This note explains whether the vector returned by `POST /features` can be stored in a Milvus
collection and used for image-to-image similarity search, and what to keep in mind when doing so.

**Short answer: yes.** The `feature_vector` field of the response is a fixed-length, L2-normalized,
dense embedding of the whole image. That is exactly the input type Milvus indexes, and cosine or
inner-product search over these vectors gives image-to-image similarity ranking.

---

## 1. What the vector is

| Property | Value |
|----------|-------|
| Name in response | `feature_vector` (array name `patch_mean_concat`) |
| Shape | `(2048,)` = `hidden_size 1024 × 2 layers` |
| Layers | 20 and 24 of the frozen DINOv3 model |
| Construction | per layer: mean over the 32×32 patch tokens after the final LayerNorm, then L2-normalize; concatenate the two layers; L2-normalize again |
| Norm | 1.0 in float32 (see the float16 note below) |
| Input | every image is EXIF-transposed, converted to RGB and resized to 512×512 (bilinear, no aspect preservation, no crop) |
| Determinism | same image, same checkpoint, same layers → same vector |

Because every image goes through the same preprocessing and the same frozen model, all vectors
live in the same space and are directly comparable with each other.

The dimension is `hidden_size × len(layers)`. With the default `layers: [20, 24]` in `config.yaml`
it is 2048. Changing `MODEL_LAYERS` changes the dimension and the meaning of the vector.

---

## 2. Reading the vector from the response

In JSON mode (the default) the vector is inline as plain floats. No base64 decoding is needed,
unlike the feature maps.

```python
import numpy as np, requests

r = requests.post(
    "http://SERVER:24500/features",
    params={"include_feature_map": "false"},
    files={"file": open("img.jpg", "rb")},
).json()

vec = np.asarray(r["feature_vector"]["data"], dtype=np.float32)   # (2048,)
assert r["feature_vector"]["shape"] == [2048]
```

Always send `include_feature_map=false` when indexing. The two feature maps add roughly 4 MB of
base64 per response and are not needed for whole-image similarity search.

In `format=npz` mode the same array is available as `data["patch_mean_concat"]`.

The response also carries the fields you need to record provenance:

- `model.checkpoint`, `model.layers`, `model.hidden_size`, `model.model_type`
- `image.filename`, `image.original_hw`, `image.exif_orientation`
- `storage_dtype`

---

## 3. Milvus collection schema

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient("http://MILVUS_HOST:19530")

schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2048)
schema.add_field("filename", DataType.VARCHAR, max_length=512)
schema.add_field("checkpoint", DataType.VARCHAR, max_length=256)
schema.add_field("layers", DataType.VARCHAR, max_length=32)       # e.g. "20,24"
schema.add_field("original_h", DataType.INT32)
schema.add_field("original_w", DataType.INT32)

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 200},
)

client.create_collection("images", schema=schema, index_params=index_params)
```

Notes on the choices:

- **`dim=2048`** must equal `feature_vector.shape[0]`. Milvus rejects inserts with a different
  dimension. The limit for `FLOAT_VECTOR` is 32768, so 2048 is well within range.
- **`metric_type`**: the vector is unit-length, so `COSINE`, `IP` and `L2` all produce the same
  ranking. Use `COSINE` (or `IP`): higher score = more similar, and the score is directly
  interpretable in `[-1, 1]`. With `L2`, lower distance = more similar.
- **`HNSW`** is the usual choice for up to a few million vectors with good recall. For very large
  collections consider `IVF_FLAT` / `IVF_SQ8`, or `DISKANN` if memory is limited. Use `FLAT` for
  exact search on small collections or to validate recall of an approximate index.
- **`FLOAT16_VECTOR`** is a valid alternative to `FLOAT_VECTOR`. The API already rounded the values
  through float16 (see section 5), so storing them as float16 in Milvus loses nothing and halves
  the vector storage.

---

## 4. Insert and search

```python
import numpy as np, requests

API = "http://SERVER:24500/features"

def embed(path: str) -> dict:
    with open(path, "rb") as f:
        r = requests.post(API, params={"include_feature_map": "false"}, files={"file": f})
    r.raise_for_status()
    return r.json()

def to_vector(r: dict) -> list[float]:
    vec = np.asarray(r["feature_vector"]["data"], dtype=np.float32)
    vec /= np.linalg.norm(vec)          # restore exact unit norm after float16 rounding
    return vec.tolist()

# insert
r = embed("img.jpg")
client.insert("images", [{
    "vector": to_vector(r),
    "filename": r["image"]["filename"],
    "checkpoint": r["model"]["checkpoint"],
    "layers": ",".join(map(str, r["model"]["layers"])),
    "original_h": r["image"]["original_hw"][0],
    "original_w": r["image"]["original_hw"][1],
}])

# search
q = embed("query.jpg")
hits = client.search(
    "images",
    data=[to_vector(q)],
    limit=10,
    output_fields=["filename"],
    search_params={"metric_type": "COSINE", "params": {"ef": 64}},
)
for hit in hits[0]:
    print(hit["distance"], hit["entity"]["filename"])   # distance is the cosine similarity
```

Use `client.insert` with lists of several hundred rows per call when bulk-indexing. The API itself
serializes inference behind a lock (one GPU, batch size 1), so parallel HTTP requests will not speed
up extraction; they only queue.

---

## 5. Things to keep in mind

### Do not mix vectors from different models in one collection

Vectors are only comparable if they were produced by the same checkpoint, the same layers and the
same input size. A different checkpoint, a different `MODEL_LAYERS`, or a different `IMAGE_SIZE`
produces a different embedding space, even when the dimension happens to be the same.

- Store `model.checkpoint` and `model.layers` with every row, or put them in the collection
  description, and check them against `GET /ready` before inserting.
- If the model or its configuration changes, create a new collection and re-embed all images.

### float16 rounding

`config.yaml` sets `storage_dtype: float16`, so the vector is rounded to float16 before being
serialized (as float32 values in JSON). Consequences:

- The norm is no longer exactly 1.0; it deviates by roughly 1e-3. This does not affect ranking in
  any meaningful way, but re-normalizing on the client (as in `to_vector` above) makes `COSINE`
  and `IP` scores identical and keeps scores directly comparable across rows.
- If you want full float32 precision, set `MODEL_STORAGE_DTYPE=float32` for the API. The vector
  is unit-norm, so it never overflows float16; the setting only affects precision.

### Cosine similarity thresholds are dataset specific

DINOv3 patch-mean features are not calibrated. A similarity of 0.9 may mean "near duplicate" for
one image domain and "loosely related" for another. Decide thresholds empirically on your own
data by looking at labelled pairs, and prefer top-k retrieval over hard thresholds when possible.

### Whole-image descriptor only

The vector is a global summary: it captures overall content, layout and texture. It is good for:

- near-duplicate detection
- "find images that look like this one"
- clustering and deduplication of a dataset

It is not good for:

- localizing or matching a small region or defect within an image
- retrieval where the object of interest occupies only a small part of the frame

For those cases use the per-patch feature maps (`layer_{L}_feature_map`, shape `(1024, 32, 32)`),
indexed as 1024 separate vectors of dimension 1024 per image per layer. Note that the feature maps
are **not** L2-normalized (unless `l2_patches: true`), so normalize them yourself before using a
cosine metric.

### Preprocessing is fixed to 512×512 without aspect preservation

Every image is stretched to a square. Two images with very different aspect ratios but the same
content will still embed close together, but extreme aspect ratios (panoramas, tall strips) are
distorted more than the model saw during training. If your data contains these, consider padding
to a square on the client before uploading.

EXIF orientation is applied before resizing, so rotated phone photos embed as displayed, not as
stored. `image.exif_orientation` in the response tells you which orientation was applied.

### Input constraints

- Accepted content types: `image/jpeg`, `image/png`, `image/bmp`, `image/webp`, `image/tiff`
  (and `application/octet-stream`, validated by decoding).
- Accepted PIL modes: `1`, `L`, `LA`, `P`, `RGB`, `RGBA`, `CMYK`. 16-bit images must be converted
  to 8-bit first, otherwise the API returns 400.
- Upload limit: `image.max_upload_mb` in `config.yaml` (default 20 MB), otherwise 413.

### Re-embed the query image with the same API

Never compute the query vector with a different pipeline (a local script, another model version).
Always send query images to the same `/features` endpoint that indexed the collection.

### Storage estimate

| Item | Size per image |
|------|----------------|
| vector as `FLOAT_VECTOR` | 8 KB |
| vector as `FLOAT16_VECTOR` | 4 KB |
| HNSW index overhead | roughly 1.2 × to 1.5 × the raw vector size |

One million images is about 8 GB of raw float32 vectors plus index, or half that with float16.

---

## 6. Quick validation checklist

1. `GET /ready` returns `hidden_size: 1024` and `layers: [20, 24]`. Record `checkpoint`.
2. `feature_vector.shape == [2048]` and `len(feature_vector.data) == 2048`.
3. `np.linalg.norm(vec)` is within 0.01 of 1.0.
4. Embedding the same file twice gives identical vectors.
5. A `FLAT`-indexed test collection returns the query image itself as the top hit with cosine
   similarity 1.0 (after re-normalization).
6. Near-duplicate pairs (same photo, different JPEG quality or slight crop) score above 0.95;
   unrelated images score clearly lower. Use this to pick thresholds for your data.
