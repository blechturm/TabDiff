#!/usr/bin/env python3
import os
import argparse
import torch
from omegaconf import OmegaConf
from .trainer import Trainer
from .utils import set_random_seed

def main(args: argparse.Namespace):
    # 1) Load the raw_config from TOML
    this_dir   = os.path.dirname(__file__)
    config_path = os.path.join(this_dir, "configs", "tabdiff_configs.toml")
    raw_config = OmegaConf.load(config_path)

    # ─────────────────────────────────────────────────────────────────────────
    # NOTE: we have REMOVED these three lines so that your
    # TOML-specified values take effect:
    #
    # raw_config["train"]["main"]["check_val_every"] = 2
    # raw_config["model_save_path"] = "debug/ckpt"
    # raw_config["result_save_path"] = "debug/result"
    # ─────────────────────────────────────────────────────────────────────────

    # 2) Merge in CLI overrides
    cli_conf = OmegaConf.from_dotlist([f"{k}={v}" for k, v in vars(args).items() if v is not None])
    config   = OmegaConf.merge(raw_config, cli_conf)

    # 3) Make sure output dirs exist
    os.makedirs(config.model_save_path, exist_ok=True)
    os.makedirs(config.result_save_path, exist_ok=True)

    # 4) Set random seed if requested
    if args.deterministic:
        set_random_seed(0)

    # 5) Launch training or sampling
    trainer = Trainer(config, args)
    if args.mode == "train":
        trainer.run_loop()
    else:
        metrics, df, _ = trainer.sample_and_report(
            ckpt_path=args.ckpt_path,
            num_samples=args.num_samples_to_generate,
            batch_size=config.sample.batch_size,
        )
        print("Final metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or sample with TabDiff")

    # --- General ---
    parser.add_argument("--dataname", type=str, required=True, help="dataset folder under data/")
    parser.add_argument("--mode",      type=str, default="train", choices=["train","sample"])
    parser.add_argument("--gpu",       type=int, default=0, help="GPU index (-1 for CPU)")
    parser.add_argument("--debug",     action="store_true", help="debug logs")
    parser.add_argument("--no_wandb",  action="store_true", help="disable wandb")
    parser.add_argument("--exp_name",  type=str, default=None, help="experiment name")
    parser.add_argument("--deterministic", action="store_true", help="fix random seed")

    # --- Training specifics ---
    parser.add_argument("--y_only",                action="store_true")
    parser.add_argument("--non_learnable_schedule",action="store_true")

    # --- Sampling specifics ---
    parser.add_argument("--num_samples_to_generate", type=int, default=None)
    parser.add_argument("--ckpt_path",               type=str, default=None)
    parser.add_argument("--report",                  action="store_true")
    parser.add_argument("--num_runs",                type=int, default=20)

    # --- Imputation (optional) ---
    parser.add_argument("--impute", action="store_true")
    parser.add_argument("--trial_start", type=int, default=0)
    parser.add_argument("--trial_size",  type=int, default=50)
    parser.add_argument("--resample_rounds", type=int, default=1)
    parser.add_argument("--impute_condition",    type=str, default="x_t")
    parser.add_argument("--y_only_model_path",   type=str, default=None)
    parser.add_argument("--w_num", type=float, default=0.6)
    parser.add_argument("--w_cat", type=float, default=0.6)

    args = parser.parse_args()

    # Device setup
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"

    main(args)
