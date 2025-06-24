#!/usr/bin/env python3
import os
import argparse
import tomli, tomli_w
from pathlib import Path
from tabdiff.main import main as tabdiff_main

def patch_toml(toml_path: Path, overrides: dict):
    txt = toml_path.read_text(encoding='utf-8')
    cfg = tomli.loads(txt)
    for key, val in overrides.items():
        parts = key.split('.')
        cur = cfg
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    new_txt = tomli_w.dumps(cfg)
    toml_path.write_text(new_txt, encoding='utf-8')

def make_args(**kwargs):
    """Turn kwargs into a minimal Namespace for tabdiff_main."""
    return argparse.Namespace(**kwargs)

def run():
    p = argparse.ArgumentParser()
    p.add_argument('--dataname',   required=True, help='data/<dataname> folder')
    p.add_argument('--ckpt-seed',  required=True, help='path to seed checkpoint .pth')
    p.add_argument('--gpu',        type=int, default=0)
    p.add_argument('--exp-seed',   default='seed_pretrain')
    p.add_argument('--exp-ft',     default='real_finetune')
    p.add_argument('--pre-steps',  type=int, default=10)
    p.add_argument('--ft-steps',   type=int, default=10)
    p.add_argument('--sample-n',   type=int, default=100000)
    p.add_argument('--no-wandb',   action='store_true')
    args = p.parse_args()

    repo = Path.cwd()
    toml_path = repo / 'tabdiff' / 'configs' / 'tabdiff_configs.toml'

    # 1) Patch TOML for seed pretrain
    seed_cfg_overrides = {
        'train.main.steps':           args.pre_steps,
        'train.main.batch_size':      2048,
        'train.main.lr':              1e-3,
        'train.main.check_val_every': args.pre_steps + 1,
        'diffusion_params.num_timesteps': 50,
        'model_save_path':            'ckpt_finetune',
        'result_save_path':           'sample_results',
        'sample.batch_size':          10000,
    }
    patch_toml(toml_path, seed_cfg_overrides)

    # 2) Run seed pre-training
    print(">>> Seed pre-training")
    seed_args = make_args(
        dataname=args.dataname,
        mode='train',
        method='tabdiff',
        gpu=args.gpu,
        debug=False,
        no_wandb=args.no_wandb,
        exp_name=args.exp_seed,
        deterministic=False
    )
    tabdiff_main(seed_args)

    # 3) Patch TOML for fine-tuning
    ft_cfg_overrides = seed_cfg_overrides.copy()
    ft_cfg_overrides.update({
        'train.main.steps':           args.ft_steps,
        'train.main.batch_size':      1024,
        'train.main.lr':              5e-4,
        'train.main.check_val_every': args.ft_steps + 1,
    })
    patch_toml(toml_path, ft_cfg_overrides)

    # 4) Run fine-tuning
    print(">>> Fine-tuning")
    ft_args = make_args(
        dataname=args.dataname,
        mode='train',
        method='tabdiff',
        gpu=args.gpu,
        debug=False,
        no_wandb=args.no_wandb,
        exp_name=args.exp_ft,
        ckpt_path=args.ckpt_seed,
        deterministic=False
    )
    tabdiff_main(ft_args)

    # 5) Run sampling + report
    print(">>> Sampling & reporting")
    sample_args = make_args(
        dataname=args.dataname,
        mode='sample',
        method='tabdiff',
        gpu=args.gpu,
        debug=False,
        no_wandb=args.no_wandb,
        exp_name=args.exp_ft,
        ckpt_path=f"ckpt_finetune/{args.exp_ft}.pth",
        num_samples_to_generate=args.sample_n,
        report=True,
        deterministic=False
    )
    tabdiff_main(sample_args)
    print("✅ Done — see ckpt_finetune/ for your CSV & logs")

if __name__ == '__main__':
    run()
