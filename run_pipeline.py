#!/usr/bin/env python3
import os
import shutil
import subprocess
import argparse
import tomli, tomli_w
from pathlib import Path

def patch_toml(toml_path: Path, overrides: dict):
    """
    Apply dotted-key → value overrides into a TOML file in-place.
    """
    text = toml_path.read_text(encoding="utf-8")
    cfg  = tomli.loads(text)
    for dotted_key, val in overrides.items():
        parts = dotted_key.split(".")
        cur   = cfg
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    toml_path.write_text(tomli_w.dumps(cfg), encoding="utf-8")

def run_cmd(cmd, cwd=None):
    """
    Print and execute a shell command.
    """
    print("> " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)

def main():
    parser = argparse.ArgumentParser(description="Full TabDiff pipeline")
    parser.add_argument("--dataname",  required=True, help="Dataset folder under data/")
    parser.add_argument("--ckpt-seed", required=True, help="Path to seed checkpoint (.pth)")
    parser.add_argument("--gpu",       type=int, default=0, help="GPU index (or -1 for CPU)")
    parser.add_argument("--exp-seed",  default="seed_pretrain", help="Name for seed experiment")
    parser.add_argument("--exp-ft",    default="real_finetune", help="Name for fine-tune experiment")
    parser.add_argument("--pre-steps", type=int, default=10, help="Epochs for pretraining")
    parser.add_argument("--ft-steps",  type=int, default=10, help="Epochs for fine-tuning")
    parser.add_argument("--sample-n",  type=int, default=100000, help="Number of samples to generate")
    parser.add_argument("--no-wandb",  action="store_true", help="Disable Weights & Biases logging")
    args = parser.parse_args()

    repo      = Path.cwd()
    toml_path = repo / "tabdiff" / "configs" / "tabdiff_configs.toml"

    # 0) Symlink in data/ and synthetic/
    for name, src in [("data","/content/data"), ("synthetic","/content/synthetic")]:
        dst = repo / name
        if dst.exists() or dst.is_symlink():
            shutil.rmtree(dst)
        os.symlink(src, dst)

    # Base TOML overrides
    base = {
        "model_save_path":                "ckpt_finetune",
        "result_save_path":               "sample_results",
        "diffusion_params.num_timesteps": 50,
        "sample.batch_size":              10000,
    }

    # ── 1) PRETRAIN ─────────────────────────────────────────────────────────
    seed = base.copy()
    seed.update({
        "train.main.steps":           args.pre_steps,
        "train.main.batch_size":      2048,
        "train.main.lr":              1e-3,
        "train.main.check_val_every": args.pre_steps + 1,
    })
    patch_toml(toml_path, seed)

    print(f">>> Seed pre-training ({args.pre_steps} epochs)")
    cmd = [
        "python3", "run_tabdiff.py",
        "--dataname", args.dataname,
        "--mode",     "train",
        "--gpu",      str(args.gpu),
        "--exp_name", args.exp_seed,
        "--debug",
    ]
    if args.no_wandb:
        cmd.append("--no_wandb")
    run_cmd(cmd, cwd=str(repo))

    # ── 2) FINETUNE ─────────────────────────────────────────────────────────
    ft = base.copy()
    ft.update({
        "train.main.steps":           args.ft_steps,
        "train.main.batch_size":      1024,
        "train.main.lr":              5e-4,
        "train.main.check_val_every": args.ft_steps + 1,
    })
    patch_toml(toml_path, ft)

    print(f">>> Fine-tuning ({args.ft_steps} epochs)")
    cmd = [
        "python3", "run_tabdiff.py",
        "--dataname",  args.dataname,
        "--mode",      "train",
        "--gpu",       str(args.gpu),
        "--exp_name",  args.exp_ft,
        "--ckpt_path", args.ckpt_seed,
        "--debug",
    ]
    if args.no_wandb:
        cmd.append("--no_wandb")
    run_cmd(cmd, cwd=str(repo))

    # ── 3) SAMPLE & REPORT ─────────────────────────────────────────────────
    print(f">>> Sampling & reporting ({args.sample_n} samples)")
    cmd = [
        "python3", "run_tabdiff.py",
        "--dataname",                args.dataname,
        "--mode",                    "sample",
        "--gpu",                     str(args.gpu),
        "--exp_name",                args.exp_ft,
        "--ckpt_path",               f"ckpt_finetune/{args.exp_ft}.pth",
        "--num_samples_to_generate", str(args.sample_n),
        "--report",
    ]
    if args.no_wandb:
        cmd.append("--no_wandb")
    run_cmd(cmd, cwd=str(repo))

    print("✅ Pipeline complete! Check ckpt_finetune/ & sample_results/")

if __name__ == "__main__":
    main()
