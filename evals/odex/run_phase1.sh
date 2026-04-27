#!/bin/bash
#
# Phase 1: Generate code solutions for ODEX instances via vLLM.
#
# Loads pre-built prompts from S3 (created by job-build-prompts.yaml),
# distributes inference across Ray workers, and uploads solutions
# to S3/MinIO for Phase 2.
#
# Prerequisites:
#   - Prompted dataset built: oc apply -f evals/odex/deploy/job-build-prompts.yaml
#   - RayCluster deployed: oc apply -f evals/odex/deploy/raycluster-code-gen.yaml
#   - vLLM server deployed: oc apply -f inference/deploy/vllm-server-deployment.yaml
#   - MinIO credentials secret configured
#   - Port-forward active: oc port-forward svc/odex-code-gen-head-svc 8265:8265
#
# Usage:
#   bash run_odex_phase1.sh
#
# Quick test with 16 instances:
#   INSTANCE_LIMIT=16 bash run_odex_phase1.sh

set -euo pipefail

# ── Configurable ────────────────────────────────────────────────
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
VLLM_ENDPOINTS="${VLLM_ENDPOINTS:-http://vllm-server:8000}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
PROMPTED_DATASET_PATH="${PROMPTED_DATASET_PATH:-s3://odex/prompts/prompted_dataset.jsonl}"
NUM_WORKERS="${NUM_WORKERS:-2}"
INSTANCE_LIMIT="${INSTANCE_LIMIT:-0}"
RUN_ID="${RUN_ID:-eval-run}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/odex-results/${RUN_ID}}"
S3_BUCKET="${S3_BUCKET:-odex}"
S3_UPLOAD_URI="${S3_UPLOAD_URI:-s3://${S3_BUCKET}/runs/${RUN_ID}/solutions.jsonl}"
TIMEOUT="${TIMEOUT:-300}"

if [[ "${DEBUG:-0}" == "1" ]]; then
  set -x
fi

# Build command args
CMD_ARGS=(
    python3 -m evals.odex.run_code_generation
    --prompted-dataset-path "${PROMPTED_DATASET_PATH}"
    --vllm-endpoints "${VLLM_ENDPOINTS}"
    --model-name-or-path "${MODEL_NAME_OR_PATH}"
    --num-workers "${NUM_WORKERS}"
    --output-path "${OUTPUT_DIR}/solutions.jsonl"
    --timeout "${TIMEOUT}"
    --instance-limit "${INSTANCE_LIMIT}"
    --s3-upload-uri "${S3_UPLOAD_URI}"
)

ray job submit \
    --address="${RAY_ADDRESS}" \
    -- "${CMD_ARGS[@]}"
