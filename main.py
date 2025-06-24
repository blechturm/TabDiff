#!/usr/bin/env python3
import os
import argparse
import torch
from omegaconf import OmegaConf
from tabdiff.trainer import Trainer

def main(args: argparse.Namespace):
    # 1) Load the user-edited TOML from tabdiff/configs/
    repo_root   = os.path.dirname(__file__)
    config_path = os.path.join(repo_root, "tabdiff", "configs", "tabdiff_configs.toml")
    raw_cfg     = OmegaConf.load(config_path)

    # ─────────────────────────────────────────────────────────────────────────
    # We no longer override `check_val_every` here—your TOML value is used.
    # ─────────────────────────────────────────────────────────────────────────

    # 2) Merge CLI args on top
    dotlist = [f"{k}={v}" for k, v in vars(args).items() if v is not None]
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    config  = OmegaConf.merge(raw_cfg, cli_cfg)

    # 3) Make sure output dirs exist
    os.makedirs(config.model_save_path,  exist_ok=True)
    os.makedirs(config.result_save_path, exist_ok=True)

    # 4) Dispatch
    trainer = Trainer(config, args)
    if args.mode == "train":
        trainer.run_loop()
    else:
        metrics, df, _ = trainer.sample_and_report(
            ckpt_path = args.ckpt_path,
            num_samples = args.num_samples_to_generate,
            batch_size = config.sample.batch_size,
        )
        print("Final metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabDiff training & sampling")

    # Required
    parser.add_argument("--dataname", type=str, required=True)
    parser.add_argument("--mode",      type=str, choices=["train","sample"], default="train")

    # Device & logging
    parser.add_argument("--gpu",       type=int, default=0)
    parser.add_argument("--debug",     action="store_true")
    parser.add_argument("--no_wandb",  action="store_true")
    parser.add_argument("--exp_name",  type=str, default=None)
    parser.add_argument("--deterministic", action="store_true")

    # Diffusion / training
    parser.add_argument("--y_only",                action="store_true")
    parser.add_argument("--non_learnable_schedule",action="store_true")

    # Sampling / evaluation
    parser.add_argument("--num_samples_to_generate", type=int, default=None)
    parser.add_argument("--ckpt_path",               type=str, default=None)
    parser.add_argument("--report",                  action="store_true")
    parser.add_argument("--num_runs",                type=int, default=20)

    # (Optional) Imputation
    parser.add_argument("--impute",           action="store_true")
    parser.add_argument("--trial_start",      type=int, default=0)
    parser.add_argument("--trial_size",       type=int, default=50)
    parser.add_argument("--resample_rounds",   type=int, default=1)
    parser.add_argument("--impute_condition", type=str, default="x_t")
    parser.add_argument("--y_only_model_path",type=str, default=None)
    parser.add_argument("--w_num",            type=float, default=0.6)
    parser.add_argument("--w_cat",            type=float, default=0.6)

    args = parser.parse_args()

    # Device setup
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"

    main(args)
