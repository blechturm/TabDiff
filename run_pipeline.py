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
    p.add_argument('--pre-steps', type=int, default=10)
    p.add_argument('--ft-steps',  type=int, default=10)
    p.add_argument('--sample-n',  type=int, default=100000)
    p.add_argument('--no-wandb',  action='store_true')
    args = p.parse_args()

    repo = Path.cwd()  # assume /content/TabDiff
    data_root = Path('/content/data')
    project_data = repo / 'data' / args.dataname

    # 0) Symlink data/ → ./data
    if (repo / 'data').exists():
        shutil.rmtree(repo / 'data')
    os.symlink(data_root, repo / 'data')

    # Copy info.json
    src_info = data_root / 'Info' / f"{args.dataname}.json"
    dst_info = project_data / 'info.json'
    dst_info.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_info, dst_info)

    # Path to toml
    toml_path = repo / 'tabdiff' / 'configs' / 'tabdiff_configs.toml'

    # 1) Patch for seed pretrain
    seed_overrides = {
        'train.main.steps':            args.pre_steps,
        'train.main.batch_size':       2048,
        'train.main.lr':               1e-3,
        'train.main.check_val_every':  args.pre_steps + 1,
        'diffusion_params.num_timesteps': 50,
        'model_save_path':             'ckpt_finetune',
        'result_save_path':            'sample_results',
        'sample.batch_size':           10000,
    }
    patch_toml(toml_path, seed_overrides)
    
    # 2) Seed pretrain via CLI
    print(">>> Seed pre-training")
    cmd = [
        sys_exe := 'python3', 'main.py',
        '--dataname', args.dataname,
        '--mode', 'train',
        '--gpu', str(args.gpu),
        '--exp_name', args.exp_seed,
    ]
    if args.no_wandb: cmd.append('--no_wandb')
    cmd.append('--debug')
    run_cmd(cmd, cwd=str(repo))

    # 3) Patch for fine-tune
    ft_overrides = seed_overrides.copy()
    ft_overrides.update({
        'train.main.steps':            args.ft_steps,
        'train.main.batch_size':       1024,
        'train.main.lr':               5e-4,
        'train.main.check_val_every':  args.ft_steps + 1,
    })
    patch_toml(toml_path, ft_overrides)

    # 4) Fine-tune via CLI
    print(">>> Fine-tuning")
    cmd = [
        sys_exe, 'main.py',
        '--dataname', args.dataname,
        '--mode', 'train',
        '--gpu', str(args.gpu),
        '--exp_name', args.exp_ft,
        '--ckpt_path', args.ckpt_seed,
    ]
    if args.no_wandb: cmd.append('--no_wandb')
    cmd.append('--debug')
    run_cmd(cmd, cwd=str(repo))

    # 5) Sample + report
    print(">>> Sampling & reporting")
    cmd = [
        sys_exe, 'main.py',
        '--dataname', args.dataname,
        '--mode', 'sample',
        '--gpu', str(args.gpu),
        '--exp_name', args.exp_ft,
        '--ckpt_path', f"ckpt_finetune/{args.exp_ft}.pth",
        '--num_samples_to_generate', str(args.sample_n),
        '--report',
    ]
    if args.no_wandb: cmd.append('--no_wandb')
    run_cmd(cmd, cwd=str(repo))

    print("✅ Done! Check `ckpt_finetune/` for checkpoints and CSV.")

if __name__ == '__main__':
    import sys
    main()
