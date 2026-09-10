#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${TRCA_RUBRIC_DIR:-${REPO_ROOT}/trca_rubrics}"

: "${TRCA_RUBRIC_API_URL:?Set the OpenAI-compatible chat-completions endpoint or /v1 base URL}"
: "${TRCA_RUBRIC_API_KEY:?Set the API key}"
: "${TRCA_RUBRIC_MODEL:?Set the API model name}"

mkdir -p "${OUTPUT_DIR}"
for benchmark in alfworld webshop searchqa; do
  "${PYTHON_BIN}" -m trca.api_generate \
    --benchmark "${benchmark}" \
    --output "${OUTPUT_DIR}/${benchmark}.json"
done
