# ODEX Evaluation Pipeline

Distributed evaluation framework for the ODEX (Open Data EXecution) benchmark using Ray + Kubernetes + vLLM.

## Overview

ODEX is a code generation benchmark that evaluates LLMs on their ability to generate Python functions from natural language descriptions and pass test cases. This evaluation pipeline:

1. **Phase 0**: Builds prompted dataset from ODEX task descriptions (one-time K8s Job)
2. **Phase 1**: Generates Python code solutions using Ray workers + vLLM
3. **Phase 2**: Executes generated code against test cases in isolated K8s Jobs

## Architecture

- **Ray**: Distributed orchestration for parallel workload processing
- **vLLM**: High-throughput LLM inference with OpenAI-compatible API
- **Kubernetes Jobs**: Isolated sandboxes for code execution
- **S3/MinIO**: Artifact storage between phases

## Prerequisites

- Kubernetes cluster with:
  - KubeRay operator installed (for RayCluster CRD)
  - Sufficient CPU/memory for Ray workers
  - RBAC permissions for creating Jobs and reading pod logs
- MinIO or S3-compatible storage
- vLLM deployment with OpenAI-compatible endpoint
- ODEX dataset in JSONL format

## Setup

### 1. Build ODEX Container Image

Build the ODEX evaluation worker image on the cluster:

```bash
oc apply -f infra/images/job-build-odex-image.yaml
oc logs -f job/build-odex-image
```

Wait for the build to complete, then verify:

```bash
oc get imagestream odex-eval
```

The image will be available at:
```
image-registry.openshift-image-registry.svc:5000/<namespace>/odex-eval:latest
```

**Note:** Update the `image:` field in all ODEX manifests if using a different registry or namespace.

See `infra/images/README.md` for alternative build methods.

### 2. Deploy MinIO (if not already deployed)

```bash
oc apply -f infra/deploy/minio.yaml
```

Create the `odex` bucket in MinIO and set up credentials:

```bash
oc create secret generic minio-credentials \
  --from-literal=MINIO_ROOT_USER=minioadmin \
  --from-literal=MINIO_ROOT_PASSWORD=minioadmin \
  --from-literal=MINIO_ENDPOINT_URL=http://minio:9000
```

### 3. Upload ODEX Dataset

Upload your ODEX dataset to S3/MinIO at `s3://odex/dataset/odex.jsonl`.

Each line should be a JSON object with:
```json
{
  "task_id": "unique-task-id",
  "intent": "Natural language description of the problem",
  "test_list": [
    {"inputs": {"arg": "value"}, "output": "expected_result"},
    ...
  ]
}
```

### 4. Deploy vLLM Server

```bash
oc apply -f inference/deploy/vllm-server-deployment.yaml
```

## Running the Pipeline

### Phase 0: Build Prompted Dataset (one-time)

Deploy the prompt-building Job:

```bash
oc apply -f evals/odex/deploy/job-build-prompts.yaml
```

Monitor the Job:

```bash
oc logs -f job/build-odex-prompts
```

This creates `s3://odex/prompts/prompted_dataset.jsonl` in MinIO.

Cleanup after completion:

```bash
oc delete job build-odex-prompts
```

### Phase 1: Generate Code Solutions

**Step 1:** Deploy the Phase 1 RayCluster:

```bash
oc apply -f evals/odex/deploy/raycluster-code-gen.yaml
```

**Step 2:** Port-forward to the Ray dashboard:

```bash
oc port-forward svc/odex-code-gen-head-svc 8265:8265
```

**Step 3:** Submit the code generation job:

```bash
# Basic usage
bash evals/odex/run_phase1.sh

# Quick test with 16 instances
INSTANCE_LIMIT=16 bash evals/odex/run_phase1.sh

# Custom configuration
MODEL_NAME_OR_PATH="Qwen/Qwen3-1.7B" \
VLLM_ENDPOINTS="http://vllm-1:8000,http://vllm-2:8000" \
NUM_WORKERS=4 \
RUN_ID=my-eval-run \
bash evals/odex/run_phase1.sh
```

Monitor in the Ray dashboard at http://localhost:8265

Results are written to `s3://odex/runs/{RUN_ID}/solutions.jsonl`

### Phase 2: Execute Tests and Grade

**Step 1:** Deploy the Phase 2 RayCluster:

```bash
oc apply -f evals/odex/deploy/rbac.yaml
oc apply -f evals/odex/deploy/raycluster-test-exec.yaml
```

**Step 2:** Port-forward to the Ray dashboard:

```bash
oc port-forward svc/odex-test-exec-head-svc 8265:8265
```

**Step 3:** Submit the test execution job:

```bash
# Basic usage (uses results from Phase 1)
RUN_ID=my-eval-run bash evals/odex/run_phase2.sh

# Quick test with 16 instances
INSTANCE_LIMIT=16 RUN_ID=my-eval-run bash evals/odex/run_phase2.sh

# Custom configuration
NUM_WORKERS=8 \
TIMEOUT=600 \
IMAGE="python:3.11-slim" \
RUN_ID=my-eval-run \
bash evals/odex/run_phase2.sh
```

Monitor in the Ray dashboard at http://localhost:8265

Results are written to:
- `s3://odex/runs/{RUN_ID}/task_results.jsonl`
- `s3://odex/runs/{RUN_ID}/aggregate_report.json`

## Configuration

### Phase 1 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RAY_ADDRESS` | Ray cluster dashboard URL | `http://127.0.0.1:8265` |
| `VLLM_ENDPOINTS` | Comma-separated vLLM endpoints | `http://vllm-server:8000` |
| `MODEL_NAME_OR_PATH` | Model identifier | `meta-llama/Llama-3.1-8B-Instruct` |
| `DATASET_PATH` | S3 URI to ODEX dataset | `s3://odex/dataset/odex.jsonl` |
| `PROMPTED_DATASET_PATH` | S3 URI to prompted dataset | `s3://odex/prompts/prompted_dataset.jsonl` |
| `NUM_WORKERS` | Number of Ray workers | `2` |
| `INSTANCE_LIMIT` | Max instances (0 = all) | `0` |
| `RUN_ID` | Unique run identifier | `eval-run` |
| `OUTPUT_DIR` | Local output directory | `/tmp/odex-results/${RUN_ID}` |
| `S3_BUCKET` | S3 bucket name | `odex` |
| `S3_UPLOAD_URI` | S3 URI for solutions | `s3://odex/runs/${RUN_ID}/solutions.jsonl` |
| `TIMEOUT` | vLLM timeout (seconds) | `300` |

### Phase 2 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RAY_ADDRESS` | Ray cluster dashboard URL | `http://127.0.0.1:8265` |
| `DATASET_PATH` | S3 URI to ODEX dataset | `s3://odex/dataset/odex.jsonl` |
| `SOLUTIONS_S3_URI` | S3 URI to solutions from Phase 1 | `s3://odex/runs/${RUN_ID}/solutions.jsonl` |
| `NUM_WORKERS` | Number of Ray workers | `4` |
| `K8S_NAMESPACE` | K8s namespace for Jobs | Auto-detect |
| `SERVICE_ACCOUNT` | K8s ServiceAccount | `odex-executor` |
| `TIMEOUT` | Per-task timeout (seconds) | `300` |
| `IMAGE` | Python container image | `python:3.11-slim` |
| `INSTANCE_LIMIT` | Max instances (0 = all) | `0` |
| `RUN_ID` | Unique run identifier | `eval-run` |
| `OUTPUT_DIR` | Local output directory | `/tmp/odex-results/${RUN_ID}` |
| `S3_BUCKET` | S3 bucket name | `odex` |
| `S3_UPLOAD_URI` | S3 URI prefix for results | `s3://odex/runs/${RUN_ID}/` |

## Results Format

### Task Results (`task_results.jsonl`)

Each line contains:

```json
{
  "task_id": "task-123",
  "passed": true,
  "total_tests": 10,
  "passed_tests": 10,
  "pass_rate": 1.0,
  "solution_exists": true,
  "error": null,
  "test_outputs": [
    {
      "test_index": 0,
      "inputs": {"x": 5},
      "expected": 25,
      "actual": 25,
      "passed": true,
      "error": null
    }
  ]
}
```

### Aggregate Report (`aggregate_report.json`)

```json
{
  "total_tasks": 100,
  "passed_tasks": 75,
  "failed_tasks": 20,
  "error_tasks": 5,
  "empty_solution_tasks": 0,
  "pass_rate": 0.75,
  "avg_test_pass_rate": 0.82,
  "passed_ids": ["task-1", "task-2", ...],
  "failed_ids": ["task-50", ...],
  "error_ids": ["task-99", ...]
}
```

## Code Structure

```
evals/odex/
├── __init__.py
├── prompt.py                  # Prompt building and code extraction
├── code_worker.py             # Ray worker for Phase 1
├── test_worker.py             # Ray worker for Phase 2
├── executor.py                # K8s Job manager for code execution
├── grader.py                  # Grading logic and result aggregation
├── build_prompt_dataset.py    # Phase 0 script
├── run_code_generation.py     # Phase 1 orchestrator
├── run_test_execution.py      # Phase 2 orchestrator
├── run_phase1.sh              # Shell wrapper for Phase 1
├── run_phase2.sh              # Shell wrapper for Phase 2
├── deploy/                    # K8s manifests
│   ├── job-build-prompts.yaml
│   ├── raycluster-code-gen.yaml
│   ├── raycluster-test-exec.yaml
│   └── rbac.yaml
└── README.md
```

## Troubleshooting

### Ray Job Submission Fails

Verify port-forward is active:
```bash
curl http://localhost:8265/api/version
```

Check Ray cluster status:
```bash
ray status --address=http://127.0.0.1:8265
```

### K8s Job Failures

List Jobs:
```bash
oc get jobs -l app=odex-eval
```

Check Job status:
```bash
oc describe job <job-name>
```

View pod logs:
```bash
oc logs -l job-name=<job-name>
```

### RBAC Errors

Verify ServiceAccount has correct permissions:
```bash
oc get role odex-executor -o yaml
oc get rolebinding odex-executor -o yaml
```

### vLLM Connection Issues

Test vLLM endpoint:
```bash
curl http://vllm-server:8000/v1/models
```

## Performance Tips

1. **Scale workers**: Increase `NUM_WORKERS` to match available vLLM endpoints (Phase 1) or desired concurrency (Phase 2)
2. **Batch size**: Ray automatically distributes instances across workers
3. **K8s resources**: Adjust CPU/memory in `executor.py` if needed (currently 500m-2 CPU, 1-2Gi RAM)
4. **Timeouts**: Increase `TIMEOUT` for complex tasks that may take longer
5. **Multiple vLLM**: Use `VLLM_ENDPOINTS="http://vllm-1:8000,http://vllm-2:8000"` for higher throughput

## Resumability

Both phases support resumability:

- **Phase 1**: Skips task IDs already in the output file. Safe to re-run after failures.
- **Phase 2**: Currently processes all instances. (Future: add resumability support)

## Related Evaluations

See also:
- `evals/swe_bench/`: SWE-bench evaluation pipeline (same architecture)
- `evals/common/`: Shared abstractions (InferenceWorker, BaseAggregateReport, s3_storage, etc.)
