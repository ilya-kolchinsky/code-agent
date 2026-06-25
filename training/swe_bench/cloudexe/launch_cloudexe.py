"""Training launcher for Cloudexe GPU servers.

Starts a local Ray cluster and submits the OpenRLHF GRPO training job
configured for single-node 8×H100 with colocated rollout/training.

Usage (from the Cloudexe base instance):
    cloudexe --gpuspec H100x8 -- /usr/bin/python3 \\
        /root/code-agent/training/swe_bench/cloudexe/launch_cloudexe.py

    # Quick test with 4 instances
    cloudexe --gpuspec H100x8 -- /usr/bin/python3 \\
        /root/code-agent/training/swe_bench/cloudexe/launch_cloudexe.py \\
        --max-samples 4

All training hyperparameters match train_swe_bench_grpo.sh; only
infrastructure flags change (single-node, local Ray, local sandbox).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch GRPO training on Cloudexe"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-9B",
        help="Model name or path (default: Qwen/Qwen3.5-9B)",
    )
    parser.add_argument(
        "--dataset",
        default="/root/swe_bench_train.jsonl",
        help="Training dataset JSONL path",
    )
    parser.add_argument(
        "--save-path",
        default="/root/checkpoints/qwen3.5-9b-grpo-swe",
        help="Checkpoint output directory",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPUs (default: 8)",
    )
    parser.add_argument(
        "--zero-stage",
        type=int,
        default=3,
        choices=[2, 3],
        help="DeepSpeed ZeRO stage (default: 3)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=40,
        help="Max training samples per episode (default: 40)",
    )
    parser.add_argument(
        "--env-map",
        default="/root/swe-env-map.json",
        help="Path to env mapping JSON from setup_local_envs.py",
    )
    parser.add_argument(
        "--repo-cache",
        default="/root/repo-cache",
        help="Path to bare repo cache from setup_local_envs.py",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Max agent steps per episode (default: 100)",
    )
    parser.add_argument(
        "--max-rollout-retries",
        type=int,
        default=2,
        help="Max retries on infrastructure failure (default: 2)",
    )
    return parser.parse_args()


def setup_environment(args):
    os.environ["HF_HOME"] = "/huggingface-public"
    os.environ["RAY_TMPDIR"] = "/tmp/ray"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TRITON_CACHE_DIR"] = "/tmp/.triton"
    os.environ["XDG_CACHE_HOME"] = "/tmp/.cache"
    os.environ["NVTX_DISABLE"] = "1"

    os.environ["SWE_ENVIRONMENT"] = "local"
    os.environ["SWE_ENV_MAP"] = args.env_map
    os.environ["SWE_REPO_CACHE"] = args.repo_cache
    os.environ["SWE_MAX_STEPS"] = str(args.max_steps)
    os.environ["SWE_MAX_ROLLOUT_RETRIES"] = str(args.max_rollout_retries)

    os.makedirs("/tmp/ray", exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)


def start_ray(num_gpus: int):
    print(f"Starting local Ray cluster with {num_gpus} GPUs...")
    subprocess.run(
        [
            "ray", "start", "--head",
            f"--num-gpus={num_gpus}",
            "--num-cpus=16",
            "--temp-dir=/tmp/ray",
        ],
        check=True,
    )
    print("Ray cluster started")


def run_training(args):
    working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    env_vars = {
        "SWE_ENVIRONMENT": "local",
        "SWE_ENV_MAP": args.env_map,
        "SWE_REPO_CACHE": args.repo_cache,
        "SWE_MAX_STEPS": str(args.max_steps),
        "SWE_MAX_ROLLOUT_RETRIES": str(args.max_rollout_retries),
    }
    env_vars_json = ",".join(
        f'"{k}": "{v}"' for k, v in env_vars.items()
    )
    runtime_env = f'{{"working_dir": "{working_dir}", "env_vars": {{{env_vars_json}}}}}'

    cmd = [
        "ray", "job", "submit",
        "--address=http://127.0.0.1:8265",
        f"--runtime-env-json={runtime_env}",
        "--",
        "python3", "-m", "openrlhf.cli.train_ppo_ray",
        # Model and checkpointing
        "--actor.model_name_or_path", args.model,
        "--ckpt.output_dir", args.save_path,
        "--ckpt.path", f"{args.save_path}/ckpt",
        "--ckpt.save_hf",
        "--ckpt.max_num", "3",
        "--ckpt.save_steps", "20",
        # Agent and data
        "--train.agent_func_path", "agent_func.py",
        "--data.prompt_dataset", args.dataset,
        "--data.input_key", "input",
        "--data.label_key", "label",
        "--data.max_len", "16384",
        "--rollout.max_new_tokens", "12288",
        # Batch sizes
        "--rollout.batch_size", "4",
        "--rollout.n_samples_per_prompt", "4",
        "--train.batch_size", "8",
        "--train.micro_batch_size", "1",
        "--rollout.micro_batch_size", "1",
        "--data.max_samples", str(args.max_samples),
        "--train.max_epochs", "1",
        "--train.num_episodes", "1",
        "--train.dynamic_batch_enable",
        # Hardware: single-node colocated
        "--actor.num_nodes", "1",
        "--actor.num_gpus_per_node", str(args.num_gpus),
        "--ref.num_nodes", "1",
        "--ref.num_gpus_per_node", str(args.num_gpus),
        "--train.colocate_all",
        "--vllm.enable_sleep",
        "--ds.enable_sleep",
        "--vllm.num_engines", "1",
        "--vllm.tensor_parallel_size", str(args.num_gpus),
        "--vllm.gpu_memory_utilization", "0.45",
        "--vllm.sync_backend", "nccl",
        "--vllm.enforce_eager",
        # DeepSpeed
        f"--ds.zero_stage", str(args.zero_stage),
        "--actor.gradient_checkpointing_enable",
        "--ds.adam_offload",
        "--ds.param_dtype", "bf16",
        # Algorithm
        "--algo.advantage.estimator", "group_norm",
        "--actor.adam.lr", "5e-7",
        "--algo.kl.init_coef", "0.01",
        "--algo.kl.use_loss",
        "--algo.kl.estimator", "k2",
        # Logging
        "--logger.tensorboard_dir", f"{args.save_path}/runs",
        "--logger.logging_steps", "1",
        "--eval.steps", "-1",
    ]

    print("Submitting training job...")
    print(f"  Model: {args.model}")
    print(f"  GPUs: {args.num_gpus} (colocated)")
    print(f"  ZeRO stage: {args.zero_stage}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Save path: {args.save_path}")

    result = subprocess.run(cmd)
    return result.returncode


def main():
    args = parse_args()

    setup_environment(args)
    start_ray(args.num_gpus)

    exit_code = run_training(args)

    subprocess.run(["ray", "stop"], capture_output=True)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
