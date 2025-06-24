#!/usr/bin/env python3
import os
import shutil
import subprocess
import argparse
import tomli, tomli_w
from pathlib import Path

def patch_toml(toml_path: Path, overrides: dict):
    txt = toml_path.read_text(encoding='utf-8')
    cfg = tomli.loads(txt)
    for key, val in overrides.items():
        parts = key.split('.')
        cur = cfg
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    toml_path.write_text(tomli_w.dumps(cfg), encoding='utf-8')

def run_cmd(cmd, cwd=None):
    print("> " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataname',  required=True, help='folder under data/, e.g. contest_flat')
    p.add_argument('--ckpt-seed', required=True, help='path to seed checkpoint .pth')
    p.add_argument('--gpu',       type=int, default=0)
    p.add_argument('--exp-seed',  default='seed_pretrain')
    p.add_argument('--exp-ft',    default='real_finetune')
    p.add_argument('--sample-n',  type=int, default=100000)
    p.add_argument('--no-wandb',  action='store_true')
    args = p.parse_args()

    repo = Path.cwd()  # expect /content/TabDiff
    data_root = Path('/content/data')
    
    # 0) Symlink /content/data → ./data for TabDiff CLI
    if (repo / 'data').exists():
        shutil.rmtree(repo / 'data')
    os.symlink(data_root, repo / 'data')
    # 0b) Symlink /content/synthetic → ./synthetic for metrics
    synth_src = Path('/content/synthetic')
    synth_dst = repo / 'synthetic'
    if synth_dst.exists() or synth_dst.is_symlink():
        shutil.rmtree(synth_dst)
    os.symlink(synth_src, synth_dst)

    # Path to the toml config
    toml_path = repo / 'tabdiff' / 'configs' / 'tabdiff_configs.toml'

    # 1) Patch for seed pretrain (disable interim eval)
    seed_overrides = {
        'train.main.steps':            10,
        'train.main.batch_size':       2048,
        'train.main.lr':               1e-3,
        'train.main.check_val_every':  9999,
        'diffusion_params.num_timesteps': 50,
        'model_save_path':             'ckpt_finetune',
        'result_save_path':            'sample_results',
        'sample.batch_size':           10000,
    }
    patch_toml(toml_path, seed_overrides)

    # 2) Seed pre-training
    print(">>> Seed pre-training")
    cmd = [
        'python3', 'main.py',
        '--dataname', args.dataname,
        '--mode', 'train',
        '--gpu', str(args.gpu),
        '--exp_name', args.exp_seed,
        '--debug'
    ]
    if args.no_wandb:
        cmd.append('--no_wandb')
    run_cmd(cmd, cwd=str(repo))

    # 3) Patch for fine-tuning (also disable interim eval)
    ft_overrides = seed_overrides.copy()
    ft_overrides.update({
        'train.main.steps':            10,
        'train.main.batch_size':       1024,
        'train.main.lr':               5e-4,
        'train.main.check_val_every':  9999,
    })
    patch_toml(toml_path, ft_overrides)

    # 4) Fine-tuning
    print(">>> Fine-tuning")
    cmd = [
        'python3', 'main.py',
        '--dataname', args.dataname,
        '--mode', 'train',
        '--gpu', str(args.gpu),
        '--exp_name', args.exp_ft,
        '--ckpt_path', args.ckpt_seed,
        '--debug'
    ]
    if args.no_wandb:
        cmd.append('--no_wandb')
    run_cmd(cmd, cwd=str(repo))

    # 5) Sampling & reporting
    print(">>> Sampling & reporting")
    cmd = [
        'python3', 'main.py',
        '--dataname', args.dataname,
        '--mode', 'sample',
        '--gpu', str(args.gpu),
        '--exp_name', args.exp_ft,
        '--ckpt_path', f"ckpt_finetune/{args.exp_ft}.pth",
        '--num_samples_to_generate', str(args.sample_n),
        '--report'
    ]
    if args.no_wandb:
        cmd.append('--no_wandb')
    run_cmd(cmd, cwd=str(repo))

    print("✅ Done! Check `ckpt_finetune/` for checkpoints and `synthetic/{args.dataname}` for metrics.")

if __name__ == '__main__':
    main()
