#!/usr/bin/env python3
import os
import argparse
import tomli, tomli_w
from tabdiff.trainer import Trainer

def load_and_patch_toml(toml_path, overrides):
    with open(toml_path,'rb') as f:
        cfg = tomli.loads(f.read())
    # apply all dotted-key overrides
    for key, val in overrides.items():
        parts = key.split('.')
        cur = cfg
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    # write back
    with open(toml_path,'wb') as f:
        f.write(tomli_w.dumps(cfg).encode())
    return cfg

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root',  default='data',      help='where data/<name>/ lives')
    p.add_argument('--dataname',   required=True)
    p.add_argument('--ckpt-seed',  required=True,       help='seed-pretrain .pth')
    p.add_argument('--gpu',        type=int, default=0)
    p.add_argument('--exp',        default='run1')
    p.add_argument('--pre-steps',  type=int, default=200)
    p.add_argument('--ft-steps',   type=int, default=50)
    p.add_argument('--batch-seed', type=int, default=2048)
    p.add_argument('--batch-ft',   type=int, default=1024)
    p.add_argument('--lr-seed',    type=float, default=1e-3)
    p.add_argument('--lr-ft',      type=float, default=5e-4)
    p.add_argument('--no-wandb',   action='store_true')
    args = p.parse_args()

    # 1) Patch TOML
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
    
    # 2) Common Trainer args
    class Args: pass
    A = Args()
    A.device     = f'cuda:{args.gpu}' if args.gpu>=0 else 'cpu'
    A.dataname   = args.dataname
    A.exp_name   = args.exp
    A.no_wandb   = args.no_wandb
    A.debug      = False  # turn True for verbose
    A.gpu        = args.gpu

    # 3) Pre-train
    print(">>> Starting seed pretrain")
    trainer = Trainer(cfg, A)
    trainer.run_loop(checkpoint_path=None)

    # 4) Fine-tune
    print(">>> Starting fine-tuning")
    A.ckpt_path = args.ckpt_seed
    overrides.update({
        'train.main.steps':       args.ft_steps,
        'train.main.batch_size':  args.batch_ft,
        'train.main.lr':          args.lr_ft,
        'train.main.check_val_every': args.ft_steps + 1,
    })
    # re-write TOML for finetune
    load_and_patch_toml(toml_path, overrides)
    trainer = Trainer(cfg, A)
    trainer.run_loop(checkpoint_path=args.ckpt_seed)

    # 5) Sample + report
    print(">>> Sampling & reporting")
    A.ckpt_path = os.path.join('ckpt_finetune', f'{args.exp}.pth')
    A.num_samples_to_generate = 100_000
    metrics, _, _ = trainer.sample_and_report(
        ckpt_path=A.ckpt_path,
        num_samples=A.num_samples_to_generate,
        batch_size=cfg['sample']['batch_size']
    )
    print("Metrics:", metrics)

if __name__ == '__main__':
    main()
