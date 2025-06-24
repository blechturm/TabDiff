#!/usr/bin/env python3
import os
import argparse
import torch
from omegaconf import OmegaConf
from tabdiff.trainer import Trainer

def main(args: argparse.Namespace):
    # 1) Load the user-edited TOML config
    repo_root   = os.path.dirname(__file__)
    cfg_path    = os.path.join(repo_root, "tabdiff", "configs", "tabdiff_configs.toml")
    raw_cfg     = OmegaConf.load(cfg_path)

    # ─────────────────────────────────────────────────────────────────────────
    # No hard-coded overrides here—your TOML values (e.g. check_val_every)
    # will be honored as written.
    # ─────────────────────────────────────────────────────────────────────────

    # 2) Merge CLI args on top of the TOML
    dotlist = [f"{k}={v}" for k, v in vars(args).items() if v is not None]
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    config  = OmegaConf.merge(raw_cfg, cli_cfg)

    # 3) Ensure output directories exist (with fallbacks)
    model_dir  = config.get("model_save_path", None) or "debug/ckpt"
    result_dir = config.get("result_save_path", None) or "debug/result"
    os.makedirs(model_dir,  exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    # Write them back into config so Trainer sees the correct paths
    config.model_save_path  = model_dir
    config.result_save_path = result_dir

    # 4) Dispatch to training or sampling
    trainer = Trainer(config, args)
    if args.mode == "train":
        trainer.run_loop()
    else:
        metrics, df, _ = trainer.sample_and_report(
            ckpt_path            = args.ckpt_path,
            num_samples          = args.num_samples_to_generate,
            batch_size           = config.sample.batch_size,
        )
        print("Final metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabDiff training & sampling")

    # Required
    parser.add_argument("--dataname", type=str, required=True, help="folder under data/")
    parser.add_argument("--mode",      type=str, choices=["train","sample"], default="train")

    # Device & logging
    parser.add_argument("--gpu",          type=int,   default=0,       help="GPU index (-1 for CPU)")
    parser.add_argument("--debug",        action="store_true",        help="Enable debug mode")
    parser.add_argument("--no_wandb",     action="store_true",        help="Disable Weights & Biases")
    parser.add_argument("--exp_name",     type=str,   default=None,    help="Experiment name")
    parser.add_argument("--deterministic",action="store_true",        help="Fix random seeds")

    # Diffusion/training specifics
    parser.add_argument("--y_only",                 action="store_true")
    parser.add_argument("--non_learnable_schedule", action="store_true")

    # Sampling/evaluation specifics
    parser.add_argument("--num_samples_to_generate", type=int,   default=None)
    parser.add_argument("--ckpt_path",               type=str,   default=None)
    parser.add_argument("--report",                  action="store_true")
    parser.add_argument("--num_runs",                type=int,   default=20)

    # (Optional) Imputation configs
    parser.add_argument("--impute",            action="store_true")
    parser.add_argument("--trial_start",       type=int,   default=0)
    parser.add_argument("--trial_size",        type=int,   default=50)
    parser.add_argument("--resample_rounds",    type=int,   default=1)
    parser.add_argument("--impute_condition",  type=str,   default="x_t")
    parser.add_argument("--y_only_model_path", type=str,   default=None)
    parser.add_argument("--w_num",             type=float, default=0.6)
    parser.add_argument("--w_cat",             type=float, default=0.6)

    args = parser.parse_args()

    # Device setup
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"

    main(args)
