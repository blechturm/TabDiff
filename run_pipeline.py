#!/usr/bin/env python3
import os
import shutil
import subprocess
import argparse
import tomli, tomli_w
from pathlib import Path

def patch_toml(toml_path: Path, overrides: dict):
    """Apply a dict of dotted-key → value overrides into a TOML file."""
    text = toml_path.read_text(encoding="utf-8")
    cfg  = tomli.loads(text)
    for key, val in overrides.items():
        parts = key.split(".")
        cur   = cfg
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    toml_path.write_text(tomli_w.dumps(cfg), encoding="utf-8")

def run_cmd(cmd, cwd=None):
    print("> " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataname",  required=True, help="dataset folder under data/")
    p.add_argument("--ckpt-seed", required=True, help="path to seed .pth checkpoint")
    p.add_argument("--gpu",       type=int, default=0)
    p.add_argument("--exp-seed",  default="seed_pretrain")
    p.add_argument("--exp-ft",    default="real_finetune")
    p.add_argument("--pre-steps", type=int, default=10,     help="epochs for pretraining")
    p.add_argument("--ft-steps",  type=int, default=10,     help="epochs for finetuning")
    p.add_argument("--sample-n",  type=int, default=100000, help="samples to generate")
    p.add_argument("--no-wandb",  action="store_true")
    args = p.parse_args()

    repo       = Path.cwd()  # e.g. /content/TabDiff
    data_root  = Path("/content/data")
    synth_src  = Path("/content/synthetic")
    toml_path  = repo / "tabdiff" / "configs" / "tabdiff_configs.toml"

    # 0) Link data and synthetic dirs into the repo for the code to see
    if (repo / "data").exists():
        shutil.rmtree(repo / "data")
    os.symlink(data_root, repo / "data")

    if (repo / "synthetic").exists() or (repo / "synthetic").is_symlink():
        shutil.rmtree(repo / "synthetic")
    os.symlink(synth_src, repo / "synthetic")

    # Base overrides from TOML
    base_over = {
        "diffusion_params.num_timesteps": 50,
        "model_save_path":                "ckpt_finetune",
        "result_save_path":               "sample_results",
        "sample.batch_size":              10000,
    }

    # 1) Patch for seed pretraining
    seed_over = base_over.copy()
    seed_over.update({
        "train.main.steps":           args.pre_steps,
        "train.main.batch_size":      2048,
        "train.main.lr":              1e-3,
        "train.main.check_val_every": args.pre_steps + 1,
    })
    patch_toml(toml_path, seed_over)

    # 2) Run seed pretraining
    print(f">>> Seed pre-training ({args.pre_steps} steps)")
    cmd = [
        "python3", "main.py",
        "--dataname", args.dataname,
        "--mode",     "train",
        "--gpu",      str(args.gpu),
        "--exp_name", args.exp_seed,
        "--debug"
    ]
    if args.no_wandb:
        cmd.append("--no_wandb")
    run_cmd(cmd, cwd=str(repo))

    # 3) Patch for fine-tuning
    ft_over = base_over.copy()
    ft_over.update({
        "train.main.steps":           args.ft_steps,
        "train.main.batch_size":      1024,
        "train.main.lr":              5e-4,
        "train.main.check_val_every": args.ft_steps + 1,
    })
    patch_toml(toml_path, ft_over)

    # 4) Run fine-tuning
    print(f">>> Fine-tuning ({args.ft_steps} steps)")
    cmd = [
        "python3", "main.py",
        "--dataname", args.dataname,
        "--mode",     "train",
        "--gpu",      str(args.gpu),
        "--exp_name", args.exp_ft,
        "--ckpt_path", args.ckpt_seed,
        "--debug"
    ]
    if args.no_wandb:
        cmd.append("--no_wandb")
    run_cmd(cmd, cwd=str(repo))

    # 5) Sampling & reporting
    print(f">>> Sampling & reporting ({args.sample_n} samples)")
    cmd = [
        "python3", "main.py",
        "--dataname",                 args.dataname,
        "--mode",                     "sample",
        "--gpu",                      str(args.gpu),
        "--exp_name",                 args.exp_ft,
        "--ckpt_path",                f"ckpt_finetune/{args.exp_ft}.pth",
        "--num_samples_to_generate",  str(args.sample_n),
        "--report"
    ]
    if args.no_wandb:
        cmd.append("--no_wandb")
    run_cmd(cmd, cwd=str(repo))

    print("✅ Pipeline complete! Check ckpt_finetune/ & sample_results/ for outputs.")

if __name__ == "__main__":
    main()
