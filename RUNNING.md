# Running the VFM Feature Extractor

Step-by-step guide to get the API running inside the `vfm` container on the GPU server and
reach it from your laptop. Commands prefixed with `host$` run on the server in MobaXterm.
Commands prefixed with `vfm$` run inside the container. Commands prefixed with `laptop$` run
on your own machine.

The API runs **inside the container** because torch, transformers, CUDA, the checkpoint and the
`cvpr27` code all live there. Port 24500 is already published by the container
(`0.0.0.0:24500->24500/tcp`), so nothing about the container has to change.

---

## 0. Before you start

You need:

- SSH access to the server (MobaXterm) with `sudo docker` rights.
- The container running: `host$ sudo docker ps` shows `vfm`.
- This repository pushed to a git remote the server can reach, **or** a way to copy files to the
  server (MobaXterm's SFTP panel on the left works).

On your laptop, commit and push first:

```
laptop$ git add -A
laptop$ git commit -m "VFM feature extractor API"
laptop$ git push
```

---

## 1. Find out where `/work` lives

Everything depends on whether `/work` inside the container is a folder from the host (a bind
mount) or only exists inside the container.

```
host$ sudo docker inspect vfm --format '{{json .Mounts}}' | python3 -m json.tool
```

Look for an entry whose `"Destination"` is `/work` (or a parent of it):

- **Found** → note its `"Source"`, for example `/home/you/work`. This is **Case A**.
- **Not found** → `/work` exists only inside the container. This is **Case B**.

Or check from inside the container. `docker exec -it vfm bash` lands you in `/work`, where
`cvpr27` already lives; the API will sit next to it as `/work/vfm-feature-extractor`.

```
vfm$ pwd                      # /work
vfm$ mount | grep " /work"
```

A line such as `/dev/sda1 on /work type ext4 ...` means `/work` comes from the host (Case A).
No output means Case B.

---

## 2. Put the code in the container

### Case A — `/work` is a bind mount (preferred)

Clone on the host directly into the mounted folder. It appears inside the container immediately
and survives container recreation.

```
host$ cd /home/you/work                      # the "Source" from step 1
host$ git clone <repo-url> vfm-feature-extractor
```

If the server has no internet, drag the project folder into `/home/you/work/` with the MobaXterm
SFTP panel instead (skip `.venv`, `__pycache__`, `.pytest_cache`, `.ruff_cache`).

### Case B — `/work` is inside the container only

Get the code onto the host first, then copy it in. It survives `docker restart` but not
`docker rm`; when the container is next recreated, add a bind mount for it.

```
host$ cd ~
host$ git clone <repo-url> vfm-feature-extractor        # or SFTP drag-and-drop
host$ sudo docker cp ~/vfm-feature-extractor vfm:/work/vfm-feature-extractor
```

### Check

```
host$ sudo docker exec vfm ls /work/vfm-feature-extractor
```

You should see `common  config.yaml  feature_extractor  run.sh  ...`.

---

## 3. Enter the container and check the environment

```
host$ sudo docker exec -it vfm bash
vfm$ cd /work/vfm-feature-extractor
```

Check the Python version (needs **3.11 or newer**) and what web packages already exist:

```
vfm$ python --version
vfm$ pip list 2>/dev/null | grep -i -E "^(pydantic|fastapi|uvicorn|starlette|python-multipart|pyyaml) "
```

- If `pydantic` is listed with a version starting with `1.`, **stop here** and check what depends
  on it before upgrading (`pip show pydantic` → "Required-by"). transformers and torch do not pin
  pydantic 1, but something else in the image might.
- If nothing is listed, that is fine; the next step installs it.

Confirm the model code and checkpoint are where `config.yaml` expects them:

```
vfm$ ls /work/cvpr27/platform/feature_export/extract.py
vfm$ ls /work/cvpr27/export_snap3/config.json
```

---

## 4. Check the model's own YAML (once)

The reference command leaves two settings to the model's YAML. The API must use the same values.
This step is **read-only on the model side**: you only read the model's YAML. If anything
differs, you edit **this app's** `config.yaml`, never the file under `cvpr27`.

```
vfm$ cat /work/cvpr27/platform/feature_export/configs/manufacturing_snap3.yaml
```

Compare with `/work/vfm-feature-extractor/config.yaml`:

| Key in the model's YAML | Expected | If different |
|-------------------------|----------|--------------|
| `l2_patches` | `false` | set `model.l2_patches` in the app's `config.yaml` to the same value |
| `amp` | `bf16` | set `model.amp` in the app's `config.yaml` to the same value |
| `resize_mode` | `resize` | tell the developer; the API always resizes to 512×512 and cannot match `native` |

To edit the app's config:

```
vfm$ nano /work/vfm-feature-extractor/config.yaml     # or vi
```

---

## 5. Install the web dependencies

Only the web layer is installed. **Do not** install or upgrade torch, numpy, pillow or
transformers; the image already has the versions the model was validated with.

```
vfm$ pip install -r requirements.txt
```

If the container has no internet, install on the host into a wheel folder and copy it in:

```
host$ pip download -r requirements.txt -d wheels        # on a machine with internet
host$ sudo docker cp wheels vfm:/tmp/wheels
vfm$ pip install --no-index --find-links /tmp/wheels -r requirements.txt
```

---

## 6. First start — in the foreground

Start it attached the first time so you see the model loading and any error directly.

```
vfm$ cd /work/vfm-feature-extractor
vfm$ bash run.sh
```

`run.sh` exports the PyTorch allocator variables the model requires, then runs
`python -m feature_extractor.main` and appends everything to `logs/uvicorn.out`. Because
of that redirect the terminal stays quiet; watch the log in a second terminal:

```
host$ sudo docker exec vfm tail -f /work/vfm-feature-extractor/logs/uvicorn.out
```

Expected lines, in order:

```
Building vfm extractor with image size 512
Loading DINOv3 checkpoint /work/cvpr27/export_snap3 on cuda:0 with amp bf16
Loading weights: 100%|██████████| 415/415
Model ready: hidden 1024, layers [20, 24], patch [16, 16]
Warmup forward done
Uvicorn running on http://0.0.0.0:24500
```

If it fails instead, the traceback is in that file. Common ones:

| Message | Fix |
|---------|-----|
| `feature_export package not found under /work/cvpr27/platform` | `model.root` in `config.yaml` is wrong |
| `local HF export required (config.json + weights)` | `model.checkpoint` is wrong or the folder is missing `config.json` |
| `CUDA was explicitly requested but is unavailable` | another process holds the GPU, or the container lost the GPU; check `nvidia-smi` |
| `ModuleNotFoundError: pydantic_settings` | step 5 did not run |
| `[Errno 98] Address already in use` | something already listens on 24500; see §10 |

Stop it with `Ctrl+C` once you have seen `Uvicorn running`.

---

## 7. Start it for real — detached

Leave the container (`vfm$ exit`) and start the service from the host so it keeps running after
you close MobaXterm:

```
host$ sudo docker exec -d vfm bash /work/vfm-feature-extractor/run.sh
```

Give it 10 to 20 seconds to load the model, then check:

```
host$ curl http://localhost:24500/health
{"server_status":"healthy"}

host$ curl http://localhost:24500/ready
{"server_status":"ready","backend":"vfm","model":{"checkpoint":"/work/cvpr27/export_snap3", ...}}
```

`/ready` returns `503 {"detail":"Model is not loaded"}` while the model is still loading. Retry.

---

## 8. Test from your laptop

Replace `SERVER` with the server's IP or hostname.

```
laptop$ curl http://SERVER:24500/ready
laptop$ curl -F "file=@some_image.jpg" "http://SERVER:24500/features?include_feature_map=false"
laptop$ curl -F "file=@some_image.jpg" "http://SERVER:24500/features?format=npz" -o out.npz
```

Verify the npz the same way the model team does:

```
laptop$ python - <<'PY'
import numpy as np
with np.load("out.npz", allow_pickle=False) as data:
    print(data["patch_mean_concat"].shape, data["patch_mean_concat"].dtype)   # (2048,) float16
    print(data["layer_24_feature_map"].shape)                                  # (1024, 32, 32)
PY
```

The interactive API docs are at `http://SERVER:24500/docs`.

If the host can reach it but the laptop cannot, the host firewall is blocking the port:

```
host$ sudo ufw status
host$ sudo ufw allow 24500/tcp        # only if ufw is active and blocking
```

---

## 9. One-time correctness check against the CLI

Run the same image through both the CLI and the API and compare the vector. Pick any image
inside the container, for example the one from the model team's example.

```
vfm$ cd /work/cvpr27
vfm$ PYTORCH_ALLOC_CONF=backend:native,expandable_segments:False \
     PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:False \
     python platform/feature_export/extract.py \
       --input data/cdfsod/DIOR/train/00001.jpg --out /tmp/cli_check \
       --checkpoint export_snap3 --layers 20 24 --features patch_mean_concat feature_map \
       --filename-mode stem --storage-dtype float16 --input-size 512 --batch-size 1 --device cuda:0

vfm$ curl -F "file=@data/cdfsod/DIOR/train/00001.jpg" "http://localhost:24500/features?format=npz" -o /tmp/api_check.npz

vfm$ python - <<'PY'
import numpy as np
cli = np.load("/tmp/cli_check/00001.npz", allow_pickle=False)
api = np.load("/tmp/api_check.npz", allow_pickle=False)
for key in ("patch_mean_concat", "layer_20_feature_map", "layer_24_feature_map"):
    a, b = cli[key].astype(np.float32), api[key].astype(np.float32)
    print(key, a.shape, "max abs diff:", np.abs(a - b).max())
PY
```

All three `max abs diff` values should be 0 or at float16 rounding level (below 1e-3).
The `--out` folder must not exist beforehand; delete `/tmp/cli_check` to rerun.

---

## 10. Day-to-day operations

### Watch the logs

```
host$ sudo docker exec vfm tail -f /work/vfm-feature-extractor/logs/vfm_feature_extractor.log
```

`vfm_feature_extractor.log` is the application log (rotating, 10 MB × 20). `uvicorn.out` is the
raw process output including startup tracebacks. Both can also be downloaded as a zip:

```
laptop$ curl http://SERVER:24500/logs -o logs.zip
```

### Is it running?

```
host$ sudo docker exec vfm pgrep -af "feature_extractor.main"
host$ sudo docker exec vfm ss -ltnp | grep 24500
```

### Stop

```
host$ sudo docker exec vfm pkill -f "feature_extractor.main"
```

### Restart

```
host$ sudo docker exec vfm pkill -f "feature_extractor.main"
host$ sudo docker exec -d vfm bash /work/vfm-feature-extractor/run.sh
```

### Update to a new version

Case A (bind mount):

```
host$ cd /home/you/work/vfm-feature-extractor && git pull
```

Case B (docker cp):

```
host$ cd ~/vfm-feature-extractor && git pull
host$ sudo docker cp ~/vfm-feature-extractor/. vfm:/work/vfm-feature-extractor/
```

Then restart as above. If `requirements.txt` changed, rerun step 5 first.

### Change a setting without editing files

Any `config.yaml` key can be overridden by an environment variable for one launch, for example
a different port or float32 output:

```
host$ sudo docker exec -d -e SERVER_PORT=24501 -e MODEL_STORAGE_DTYPE=float32 vfm bash /work/vfm-feature-extractor/run.sh
```

The port must be one the container publishes (24500 to 24509).

### After the container restarts

The API does **not** start on its own when the container restarts. Run step 7 again.

---

## 11. Quick reference

```
host$ sudo docker exec -it vfm bash                                      # enter
host$ sudo docker exec -d vfm bash /work/vfm-feature-extractor/run.sh    # start
host$ sudo docker exec vfm pkill -f "feature_extractor.main"             # stop
host$ sudo docker exec vfm tail -f /work/vfm-feature-extractor/logs/vfm_feature_extractor.log
host$ curl http://localhost:24500/ready                                  # check
```
