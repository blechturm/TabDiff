#!/usr/bin/env python3
import torch
import argparse
from tabdiff.main import main as tabdiff_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training or sampling with TabDiff")

    # Data & mode
    parser.add_argument(
        "--dataname", type=str, required=True,
        help="Name of the dataset folder under data/"
    )
    parser.add_argument(
        "--mode", type=str, choices=["train","sample"],
        default="train", help="train or sample"
    )

    # Device & logging
    parser.add_argument("--gpu",          type=int,   default=0,       help="GPU index (-1 for CPU)")
    parser.add_argument("--debug",        action="store_true",        help="Enable debug logging")
    parser.add_argument("--no_wandb",     action="store_true",        help="Disable WandB")
    parser.add_argument("--exp_name",     type=str,   default=None,    help="Experiment name")
    parser.add_argument("--deterministic",action="store_true",        help="Fix random seeds")

    # Model / training options
    parser.add_argument("--y_only",                 action="store_true")
    parser.add_argument("--non_learnable_schedule", action="store_true")

    # Sampling / evaluation
    parser.add_argument("--num_samples_to_generate", type=int,   default=None)
    parser.add_argument("--ckpt_path",               type=str,   default=None)
    parser.add_argument("--report",                  action="store_true")
    parser.add_argument("--num_runs",                type=int,   default=20)

    # Imputation (optional)
    parser.add_argument("--impute",            action="store_true")
    parser.add_argument("--trial_start",       type=int,   default=0)
    parser.add_argument("--trial_size",        type=int,   default=50)
    parser.add_argument("--resample_rounds",    type=int,   default=1)
    parser.add_argument("--impute_condition",  type=str,   default="x_t")
    parser.add_argument("--y_only_model_path", type=str,   default=None)
    parser.add_argument("--w_num",             type=float, default=0.6)
    parser.add_argument("--w_cat",             type=float, default=0.6)

    args = parser.parse_args()

    # Map GPU flag to device
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"

    # Call into the real entrypoint
    tabdiff_main(args)
