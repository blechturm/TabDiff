#!/usr/bin/env python
import os
import pickle
import torch
import argparse

from tabdiff.main import main as _tabdiff_main
from tabdiff.trainer import Trainer

def main(args):
    raw_config = args.cfg  # assume you load your patched TOML into args.cfg

    # …insert here the “comment out check_val_every” patch if needed…
    # but if you’ve already baked that into your fork’s tabdiff/main.py, skip here.

    if args.mode == 'train':
        if args.ckpt_path:
            print(f"▶▶▶ [DEBUG] Loading checkpoint: {args.ckpt_path}")
        Trainer(raw_config, args).run_loop()

    elif args.mode == 'sample':
        # ensure a valid result path
        result_path = raw_config.get('result_save_path') or 'sample_results'
        os.makedirs(result_path, exist_ok=True)

        trainer = Trainer(raw_config, args)
        results = trainer.sample_and_report(
            ckpt_path=args.ckpt_path,
            num_samples=args.num_samples_to_generate,
            batch_size=raw_config['sample']['batch_size'],
        )
        with open(os.path.join(result_path, 'config.pkl'), 'wb') as f:
            pickle.dump((raw_config, results), f)

    else:
        raise ValueError(f"Unknown mode {args.mode}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', required=True)
    parser.add_argument('--mode', choices=['train','sample'], required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--num_samples_to_generate', type=int)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--report', action='store_true')
    # and any other flags you originally needed…

    # Load your patched TOML into args.cfg here:
    import tomli
    args = parser.parse_args()
    cfg_text = open(f"TabDiff/tabdiff/configs/tabdiff_configs.toml","rb").read()
    args.cfg = tomli.loads(cfg_text)

    # set device
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    main(args)
