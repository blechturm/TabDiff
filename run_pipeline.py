#!/usr/bin/env python3
import os, shutil
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
    toml_path.write_text(tomli_w.dumps(cfg), encoding='utf-8')

def make_args(**kwargs):
    return argparse.Namespace(**kwargs)

def run():
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

    repo = Path.cwd()  # /content/TabDiff
    data_root = Path('/content/data')
    project_data = repo / 'data' / args.dataname

    # 0) Prepare data path for TabDiff
    # Symlink the entire /content/data into ./data
    if (repo / 'data').exists():
        shutil.rmtree(repo / 'data')
    os.symlink(data_root, repo / 'data')

    # Copy info JSON into data/<dataname>/info.json
    src_info = data_root / 'Info' / f"{args.dataname}.json"
    dst_info = project_data / 'info.json'
    dst_info.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_info, dst_info)

    # 1) Patch TOML for seed pretrain
    toml_path = repo / 'tabdiff' / 'configs' / 'tabdiff_configs.toml'
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

    # 2) Seed pre-training
    print(">>> Seed pre-training")
    seed_args = make_args(
        dataname      = args.dataname,
        mode          = 'train',
        method        = 'tabdiff',
        gpu           = args.gpu,
        debug         = False,
        no_wandb      = args.no_wandb,
        exp_name      = args.exp_seed,
        ckpt_path     = None,
        deterministic = False
    )
    seed_args.device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    tabdiff_main(seed_args)

    # 3) Patch TOML for fine-tuning
    ft_overrides = seed_overrides.copy()
    ft_overrides.update({
        'train.main.steps':            args.ft_steps,
        'train.main.batch_size':       1024,
        'train.main.lr':               5e-4,
        'train.main.check_val_every':  args.ft_steps + 1,
    })
    patch_toml(toml_path, ft_overrides)

    # 4) Fine-tuning
    print(">>> Fine-tuning")
    ft_args = make_args(
        dataname      = args.dataname,
        mode          = 'train',
        method        = 'tabdiff',
        gpu           = args.gpu,
        debug         = False,
        no_wandb      = args.no_wandb,
        exp_name      = args.exp_ft,
        ckpt_path     = args.ckpt_seed,
        deterministic = False
    )
    ft_args.device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    tabdiff_main(ft_args)

    # 5) Sampling & reporting
    print(">>> Sampling & reporting")
    sample_args = make_args(
        dataname                = args.dataname,
        mode                    = 'sample',
        method                  = 'tabdiff',
        gpu                     = args.gpu,
        debug                   = False,
        no_wandb                = args.no_wandb,
        exp_name                = args.exp_ft,
        ckpt_path               = f"ckpt_finetune/{args.exp_ft}.pth",
        num_samples_to_generate = args.sample_n,
        report                  = True,
        deterministic           = False
    )
    sample_args.device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    tabdiff_main(sample_args)

    print("✅ Done! Check `ckpt_finetune/` for your checkpoints and generated CSV.")

if __name__ == '__main__':
    run()
