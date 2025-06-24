import os
import pickle
import torch
import argparse

from tabdiff.trainer import Trainer


def main(args):
    # ---------------------------------------------------------------------
    # Load and merge configuration (Hydra / OmegaConf)
    # ---------------------------------------------------------------------
    raw_config = ...  # existing config loading

    # ---------------------------------------------------------------------
    # Patch out the hard-coded validation frequency override
    # (original code did: raw_config['train']['main']['check_val_every'] = 2)
    # We simply remove/comment it so user-specified values stick.
    # ---------------------------------------------------------------------
    # raw_config['train']['main']['check_val_every'] = 2

    # ---------------------------------------------------------------------
    # Initialize trainer
    # ---------------------------------------------------------------------
    trainer = Trainer(raw_config, args)

    if args.mode == 'train':
        ckpt_path = args.ckpt_path
        if ckpt_path:
            print(f"▶▶▶ [DEBUG] Loaded checkpoint: {ckpt_path}")
        trainer.run_loop()

    elif args.mode == 'sample':
        # In sample mode, ensure a non-null save path for config.pkl
        config_save_path = raw_config.get('result_save_path')
        if config_save_path is None:
            config_save_path = 'sample_results'
        os.makedirs(config_save_path, exist_ok=True)

        # Proceed with sampling
        results = trainer.sample_and_report(
            ckpt_path=args.ckpt_path,
            num_samples=args.num_samples_to_generate,
            batch_size=raw_config['sample']['batch_size'],
        )

        # Dump the config & results for reproducibility
        with open(os.path.join(config_save_path, 'config.pkl'), 'wb') as f:
            pickle.dump((raw_config, results), f)

    else:
        raise ValueError(f"Unrecognized mode: {args.mode}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training or sampling with TabDiff')
    # [.. existing parser.add_argument definitions ..]
    args = parser.parse_args()

    # Set device
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    main(args)
