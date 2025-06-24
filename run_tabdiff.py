#!/usr/bin/env python3
import torch
import argparse
from tabdiff.main import main as tabdiff_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wrapper to invoke TabDiff’s own CLI entrypoint"
    )

    # Dataset + mode
    parser.add_argument(
        "--dataname", required=True,
        help="Name of the dataset folder under data/"
    )
    parser.add_argument(
        "--mode", choices=["train","sample"], required=True,
        help="train or sample"
    )

    # Device + logging
    parser.add_argument("--gpu",        type=int,   default=0,        help="GPU index (or -1 for CPU)")
    parser.add_argument("--debug",      action="store_true",         help="Verbose debug mode")
    parser.add_argument("--no_wandb",   action="store_true",         help="Disable Weights & Biases")
    parser.add_argument("--exp_name",   type=str,   default=None,     help="Experiment name (checkpoints & logs)")

    # Training-specific
    parser.add_argument("--ckpt_path",  type=str,   default=None,     help="Checkpoint path (for finetune or sampling)")
    
    # Sampling-specific
    parser.add_argument(
        "--num_samples_to_generate", type=int, default=None,
        help="When mode=sample: how many rows to generate"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="When sampling: run multiple trials and report mean±std"
    )
    parser.add_argument(
        "--num_runs", type=int, default=20,
        help="Number of runs for --report"
    )

    # (You can add any other flags that tabdiff.main expects—e.g. imputation flags, y_only, etc.)

    args = parser.parse_args()

    # Map GPU flag → device string
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"

    # Call the repo’s CLI entrypoint
    tabdiff_main(args)
