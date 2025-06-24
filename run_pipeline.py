#!/usr/bin/env python3
import os
import argparse
import tomli, tomli_w
from tabdiff.trainer import Trainer

def load_and_patch_toml(toml_path, overrides):
    # Open in text mode so tomli.loads gets a str
    with open(toml_path, 'r', encoding='utf-8') as f:
        txt = f.read()
    cfg = tomli.loads(txt)
    # apply dotted-key overrides
    for key, val in overrides.items():
        parts = key.split('.')
        cur = cfg
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    # write back
    new_txt = tomli_w.dumps(cfg)
    with open(toml_path, 'w', encoding='utf-8') as f:
        f.write(new_txt)
    return cfg

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='data', help='where data/<name>/ lives')
    p.add_argument('--dataname', required=True)
    p.add_argument('--ckpt-seed', required=True, help='path to seed checkpoint .pth')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--exp', default='run1')
    p.add_argument('--pre-steps', type=int, default=10)
    p.add_argument('--ft-steps', type=int, default=10)
    p.add_argument('--batch-seed', type=int, default=2048)
    p.add_argument('--batch-ft', type=int, default=1024)
    p.add_argument('--lr-seed', type=float, default=1e-3)
    p.add_argument('--lr-ft', type=float, default=5e-4)
    p.add_argument('--no-wandb', action='store_true')
    args = p.parse_args()

    # 1) Patch the TOML
    toml_path = os.path.join(os.getcwd(), 'tabdiff', 'configs', 'tabdiff_configs.toml')
    overrides = {
        'train.main.steps':           args.pre_steps,
        'train.main.batch_size':      args.batch_seed,
        'train.main.lr':              args.lr_seed,
        'train.main.check_val_every': args.pre_steps + 1,
        'diffusion_params.num_timesteps': 50,
        'model_save_path':            'ckpt_finetune',
        'result_save_path':           'sample_results',
        'sample.batch_size':          10000,
    }
    cfg = load_and_patch_toml(toml_path, overrides)

    # 2) Prepare Trainer args container
    class A: pass
    A.dataname   = args.dataname
    A.exp_name   = args.exp
    A.device     = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    A.gpu        = args.gpu
    A.no_wandb   = args.no_wandb
    A.debug      = False

    # 3) Seed pre-train
    print(">>> Seed pre-training")
    trainer = Trainer(cfg, A)
    trainer.run_loop()

    # 4) Fine-tune
    print(">>> Fine-tuning")
    # rewrite TOML for fine-tune
    overrides.update({
        'train.main.steps':           args.ft_steps,
        'train.main.batch_size':      args.batch_ft,
        'train.main.lr':              args.lr_ft,
        'train.main.check_val_every': args.ft_steps + 1,
    })
    cfg = load_and_patch_toml(toml_path, overrides)
    A.ckpt_path = args.ckpt_seed
    trainer = Trainer(cfg, A)
    trainer.run_loop()

    # 5) Sample + report
    print(">>> Sampling 100k")
    A.ckpt_path = os.path.join('ckpt_finetune', f"{args.exp}.pth")
    metrics, _, _ = trainer.sample_and_report(
        ckpt_path=A.ckpt_path,
        num_samples=100_000,
        batch_size=cfg['sample']['batch_size']
    )
    print("Metrics:", metrics)

if __name__ == '__main__':
    main()
