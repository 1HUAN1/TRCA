#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/data/zh/gigpo+fcsr_verl}"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda/envs/verl-agent-retriever/bin/python}"
RETRIEVE_DATA_ROOT="${RETRIEVE_DATA_ROOT:-/root/data/mingjuchen/EMNLP2026/Benchmark_all/AgentGym/agentenv-searchqa/retrieve_data}"
INDEX_FILE="${INDEX_FILE:-$RETRIEVE_DATA_ROOT/e5_Flat.index}"
CORPUS_FILE="${CORPUS_FILE:-$RETRIEVE_DATA_ROOT/wiki-18.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-$RETRIEVE_DATA_ROOT/e5-base-v2}"
RETRIEVER_CACHE_ROOT="${RETRIEVER_CACHE_ROOT:-$PROJECT_ROOT/.cache/search_retriever}"
RETRIEVER_DEVICE="${RETRIEVER_DEVICE:-cuda}"
PORT="${PORT:-8000}"
TOPK="${TOPK:-3}"
FAISS_GPU="${FAISS_GPU:-1}"
RETRIEVER_CUDA_VISIBLE_DEVICES="${RETRIEVER_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
FAISS_GPU_USE_FLOAT16="${FAISS_GPU_USE_FLOAT16:-true}"
FAISS_GPU_SHARD="${FAISS_GPU_SHARD:-true}"

cd "$PROJECT_ROOT"

export HF_HOME="${HF_HOME:-$RETRIEVER_CACHE_ROOT/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

if [[ "$PYTHON_BIN" == "/root/data/conda/envs/verl-agent-retriever/bin/python" ]]; then
  unset PYTHONPATH LD_LIBRARY_PATH
fi

test -x "$PYTHON_BIN" || { echo "[error] missing Python: $PYTHON_BIN" >&2; exit 1; }
test -s "$INDEX_FILE" || { echo "[error] missing FAISS index: $INDEX_FILE" >&2; exit 1; }
test -s "$CORPUS_FILE" || { echo "[error] missing retrieval corpus: $CORPUS_FILE" >&2; exit 1; }
test -d "$RETRIEVER_MODEL" || { echo "[error] missing local retriever model: $RETRIEVER_MODEL" >&2; exit 1; }

ARGS=(
  "$PYTHON_BIN" examples/search/retriever/retrieval_server.py
  --index_path "$INDEX_FILE"
  --corpus_path "$CORPUS_FILE"
  --topk "$TOPK"
  --retriever_name e5
  --retriever_model "$RETRIEVER_MODEL"
  --device "$RETRIEVER_DEVICE"
  --port "$PORT"
)

if [[ "$RETRIEVER_DEVICE" == "cuda" ]]; then
  export CUDA_VISIBLE_DEVICES="$RETRIEVER_CUDA_VISIBLE_DEVICES"
  if [[ "$FAISS_GPU" == "1" ]]; then
    IFS=',' read -r -a RETRIEVER_GPU_IDS <<< "$RETRIEVER_CUDA_VISIBLE_DEVICES"
    export FAISS_GPU_EXPECTED_COUNT="${FAISS_GPU_EXPECTED_COUNT:-${#RETRIEVER_GPU_IDS[@]}}"
    export FAISS_GPU_USE_FLOAT16 FAISS_GPU_SHARD FAISS_GPU_EXPECTED_COUNT
    ARGS+=(--faiss_gpu)
  fi
elif [[ "$RETRIEVER_DEVICE" != "cpu" && "$RETRIEVER_DEVICE" != "auto" ]]; then
  echo "[error] RETRIEVER_DEVICE must be cuda, cpu, or auto" >&2
  exit 2
fi

echo "RETRIEVER_DEVICE=$RETRIEVER_DEVICE CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<cpu>}"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "FAISS_GPU=$FAISS_GPU FAISS_GPU_EXPECTED_COUNT=${FAISS_GPU_EXPECTED_COUNT:-0}"
echo "FAISS_GPU_USE_FLOAT16=$FAISS_GPU_USE_FLOAT16 FAISS_GPU_SHARD=$FAISS_GPU_SHARD"
echo "INDEX_FILE=$INDEX_FILE"
echo "CORPUS_FILE=$CORPUS_FILE"
echo "RETRIEVER_MODEL=$RETRIEVER_MODEL PORT=$PORT TOPK=$TOPK"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
exec "${ARGS[@]}"
