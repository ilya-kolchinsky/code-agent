#!/bin/bash
#
# Phase 2: Run ODEX test execution via K8s Jobs.
#
# Reads solutions from S3/MinIO (output of Phase 1), distributes test
# execution across Ray workers. Each worker creates K8s Jobs with Python
# containers to execute code against test cases. Results are graded,
# aggregated, and uploaded to S3/MinIO.
#
# Prerequisites:
#   - RayCluster deployed: oc apply -f evals/odex/deploy/raycluster-test-exec.yaml
#   - RBAC applied: oc apply -f evals/odex/deploy/rbac.yaml
#   - solutions.jsonl from Phase 1 (in S3)
#   - MinIO credentials secret (same as Phase 1)
#   - Port-forward active: oc port-forward svc/odex-test-exec-head-svc 8265:8265
#
# Usage:
#   bash run_odex_phase2.sh
#
# Quick test with 16 instances:
#   INSTANCE_LIMIT=16 bash run_odex_phase2.sh

set -euo pipefail

# ── Configurable ────────────────────────────────────────────────
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
DATASET_PATH="${DATASET_PATH:-s3://odex/dataset/odex.jsonl}"
RUN_ID="${RUN_ID:-eval-run}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/odex-results/${RUN_ID}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
K8S_NAMESPACE="${K8S_NAMESPACE:-}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-odex-executor}"
TIMEOUT="${TIMEOUT:-300}"
IMAGE="${IMAGE:-python:3.11-slim}"
INSTANCE_LIMIT="${INSTANCE_LIMIT:-0}"
S3_BUCKET="${S3_BUCKET:-odex}"
SOLUTIONS_S3_URI="${SOLUTIONS_S3_URI:-s3://${S3_BUCKET}/runs/${RUN_ID}/solutions.jsonl}"
S3_UPLOAD_URI="${S3_UPLOAD_URI:-s3://${S3_BUCKET}/runs/${RUN_ID}/}"

if [[ "${DEBUG:-0}" == "1" ]]; then
  set -x
fi

# Build command args
CMD_ARGS=(
    python3 -m evals.odex.run_test_execution
    --dataset-path "${DATASET_PATH}"
    --solutions-s3-uri "${SOLUTIONS_S3_URI}"
    --num-workers "${NUM_WORKERS}"
    --service-account "${SERVICE_ACCOUNT}"
    --timeout "${TIMEOUT}"
    --image "${IMAGE}"
    --output-dir "${OUTPUT_DIR}"
    --run-id "${RUN_ID}"
    --instance-limit "${INSTANCE_LIMIT}"
    --s3-upload-uri "${S3_UPLOAD_URI}"
)

# Only pass --k8s-namespace if explicitly set (otherwise auto-detected in-cluster)
if [[ -n "${K8S_NAMESPACE}" ]]; then
    CMD_ARGS+=(--k8s-namespace "${K8S_NAMESPACE}")
fi

ray job submit \
    --address="${RAY_ADDRESS}" \
    -- "${CMD_ARGS[@]}"
