#!/usr/bin/env python3
import torch
import argparse
from tabdiff.main import main as tabdiff_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wrapper for TabDiff CLI")

    # ── General configs ──────────────────────────────────────────────────────
    parser.add_argument(
        "--dataname", type=str, required=True,
        help="Dataset name; folder under data/"
    )
    parser.add_argument(
        "--mode", choices=["train", "sample"], required=True,
        help="train or sample"
    )
    parser.add_argument(
        "--method", type=str, default="tabdiff",
        help="Model to run; currently only 'tabdiff'"
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU index (-1 for CPU)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable verbose debug logging"
    )
    parser.add_argument(
        "--no_wandb", action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--exp_name", type=str, default=None,
        help="Experiment name (used for checkpoint & log dirs)"
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Fix random seeds for reproducibility"
    )

    # ── Diffusion/training configs ──────────────────────────────────────────
    parser.add_argument(
        "--y_only", action="store_true",
        help="Train guidance model using only the target column"
    )
    parser.add_argument(
        "--non_learnable_schedule", action="store_true",
        help="Disable learnable noise schedule"
    )

    # ── Sampling/evaluation configs ────────────────────────────────────────
    parser.add_argument(
        "--num_samples_to_generate", type=int, default=None,
        help="Number of synthetic samples to generate"
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None,
        help="Path to .pth checkpoint (for sampling or fine-tuning)"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="In sample mode: run multiple trials and report mean±std"
    )
    parser.add_argument(
        "--num_runs", type=int, default=20,
        help="Number of runs for --report mode"
    )

    # ── Imputation configs (optional) ──────────────────────────────────────
    parser.add_argument("--impute", action="store_true")
    parser.add_argument("--trial_start", type=int, default=0)
    parser.add_argument("--trial_size", type=int, default=50)
    parser.add_argument("--resample_rounds", type=int, default=1)
    parser.add_argument(
        "--impute_condition", type=str, default="x_t",
        help="Conditioning variable for imputation"
    )
    parser.add_argument(
        "--y_only_model_path", type=str, default=None,
        help="Path to unconditional guidance model checkpoint"
    )
    parser.add_argument("--w_num", type=float, default=0.6)
    parser.add_argument("--w_cat", type=float, default=0.6)

    args = parser.parse_args()

    # ── Device setup ────────────────────────────────────────────────────────
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"

    # ── Delegate to TabDiff’s main entrypoint ───────────────────────────────
    tabdiff_main(args)
