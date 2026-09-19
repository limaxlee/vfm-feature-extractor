#!/usr/bin/env bash
# Launch the VFM feature extraction API from the repo root (inside the vfm container).
#   bash run.sh                      # uses ./config.yaml
#   bash run.sh --config other.yaml  # any extra args are forwarded to the app
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

# Allocator settings must be present before torch is imported.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-backend:native,expandable_segments:False}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-backend:native,expandable_segments:False}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"

mkdir -p logs
exec python -m feature_extractor.main "$@" >> logs/uvicorn.out 2>&1
