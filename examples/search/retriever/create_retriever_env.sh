#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/root/data/conda/miniconda3/bin/conda}"
ENV_PREFIX="${ENV_PREFIX:-/root/data/conda/envs/verl-agent-retriever}"
EXPECTED_GPUS="${EXPECTED_GPUS:-8}"
REBUILD="${REBUILD:-0}"

test -x "$CONDA_BIN" || { echo "[error] conda not found: $CONDA_BIN" >&2; exit 1; }

verify_env() {
  "$ENV_PREFIX/bin/python" - "$EXPECTED_GPUS" <<'PY'
import sys
import faiss
import torch

expected = int(sys.argv[1])
missing = [
    name
    for name in ("GpuMultipleClonerOptions", "StandardGpuResources", "index_cpu_to_all_gpus", "get_num_gpus")
    if not hasattr(faiss, name)
]
detected = int(faiss.get_num_gpus()) if hasattr(faiss, "get_num_gpus") else 0
if missing or detected != expected:
    raise SystemExit(f"GPU FAISS verification failed: detected={detected}, expected={expected}, missing={missing}")
if not torch.__version__.startswith("2.6.0") or not torch.cuda.is_available():
    raise SystemExit(f"PyTorch GPU verification failed: torch={torch.__version__}, cuda={torch.version.cuda}")
print(
    f"faiss={getattr(faiss, '__version__', 'unknown')} detected_gpus={detected} "
    f"torch={torch.__version__} cuda={torch.version.cuda} missing_apis={missing}"
)
PY
}

if [[ -x "$ENV_PREFIX/bin/python" ]]; then
  if verify_env; then
    echo "[ok] reusing verified retriever environment: $ENV_PREFIX"
    exit 0
  fi
  [[ "$REBUILD" == "1" ]] || {
    echo "[error] existing retriever environment failed GPU FAISS verification; rerun with REBUILD=1" >&2
    exit 1
  }
fi

if [[ -e "$ENV_PREFIX" ]]; then
  BACKUP_PREFIX="${ENV_PREFIX}.incomplete.$(date +%Y%m%d_%H%M%S)"
  echo "[info] preserving incomplete environment at $BACKUP_PREFIX"
  mv "$ENV_PREFIX" "$BACKUP_PREFIX"
fi

export CONDA_NO_PLUGINS=true
"$CONDA_BIN" create --solver=classic --prefix "$ENV_PREFIX" python=3.10 numpy=1.26.4 pip -y
"$CONDA_BIN" install --solver=classic --prefix "$ENV_PREFIX" faiss-gpu=1.8.0 -c pytorch -c nvidia -y
"$ENV_PREFIX/bin/pip" install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
"$ENV_PREFIX/bin/pip" install transformers datasets pyserini huggingface_hub uvicorn fastapi

verify_env
echo "[ok] retriever environment ready: $ENV_PREFIX"
