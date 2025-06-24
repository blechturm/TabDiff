import glob
import json
import os
import pickle
import random

import numpy as np
from tabdiff.metrics import TabMetrics
from tabdiff.modules.main_modules import UniModMLP
from tabdiff.modules.main_modules import Model
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from tabdiff.trainer import Trainer
import src
import torch

from torch.utils.data import DataLoader
import argparse
import warnings

import wandb

from copy import deepcopy

from utils_train import TabDiffDataset

warnings.filterwarnings('ignore')


def main(args):
    device = args.device

    ## Disable scientific numerical format
    np.set_printoptions(suppress=True)
    torch.set_printoptions(sci_mode=False)

    ## Get data info
    dataname = args.dataname
    data_dir = f'data/{dataname}'
    info_path = f'data/{dataname}/info.json'
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    ## Set up flags
    is_dcr = 'dcr' in dataname

    ## Set experiment name
    exp_name = args.exp_name
    if exp_name is None:
        exp_name = 'non_learnable_schedule' if args.non_learnable_schedule else 'learnable_schedule'
    exp_name += '_y_only' if args.y_only else ''
    
    ## Load configs
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = f'{curr_dir}/configs/tabdiff_configs.toml'
    raw_config = src.load_config(config_path)
    
    print(f"{args.mode.capitalize()} Mode is Enabled")
    num_samples_to_generate = None
    ckpt_path = None
    if args.mode == 'train':
        print("NEW training is started")
    elif args.mode == 'test':
        num_samples_to_generate = args.num_samples_to_generate
        ckpt_path = args.ckpt_path
        if ckpt_path is None:
            ckpt_parent_path = f"{curr_dir}/ckpt/{dataname}/{exp_name}"
            arr = glob.glob(f"{ckpt_parent_path}/best_ema_model*")
            assert arr, f"Cannot infer ckpt_path from {ckpt_parent_path}"
            ckpt_path = arr[0]
        config_path = os.path.join(os.path.dirname(ckpt_path), 'config.pkl')
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                raw_config = pickle.load(f)
                print(f"Loaded cached config from {config_path}")

    # Create save directories
    if args.mode == 'train':
        model_save_path = 'debug/ckpt' if args.debug else f'{curr_dir}/ckpt/{dataname}/{exp_name}'
        result_save_path = model_save_path.replace('ckpt', 'result')
    else:
        if args.report:
            result_save_path = f"eval/report_runs/{exp_name}/{dataname}"
        else:
            result_save_path = os.path.dirname(ckpt_path).replace('ckpt', 'result')
    raw_config['model_save_path'] = model_save_path
    raw_config['result_save_path'] = result_save_path
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(result_save_path, exist_ok=True)

    # Deterministic
    raw_config['deterministic'] = args.deterministic
    if args.deterministic:
        print("DETERMINISTIC MODE is enabled!!!")
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        os.environ['PYTHONHASHSEED'] = '0'
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Debug overrides
    if args.debug:
        raw_config['train']['main']['check_val_every']   = 2
        raw_config['diffusion_params']['num_timesteps']  = 4
        raw_config['train']['main']['batch_size']       = 4096
        raw_config['sample']['batch_size']              = 10000

    # Prepare data loaders
    train_data = TabDiffDataset(...)
    train_loader = DataLoader(train_data, batch_size=raw_config['train']['main']['batch_size'], shuffle=True)
    val_data = TabDiffDataset(...)

    # Metrics
    metrics = TabMetrics(...)

    # Model + diffusion setup
    backbone = UniModMLP(**raw_config['unimodmlp_params'])
    model    = Model(backbone, **raw_config['diffusion_params']['edm_params'])
    diffusion= UnifiedCtimeDiffusion(...)

    # Logger
    logger = wandb.init(
        project=f"tabdiff_{dataname}",
        name=exp_name,
        config=raw_config,
        mode='disabled' if args.debug or args.no_wandb else 'online'
    )

    # Instantiate Trainer with explicit signature
    trainer = Trainer(
        diffusion,
        train_loader,
        train_data,
        val_data,
        metrics,
        logger,
        lr=raw_config['train']['main']['lr'],
        weight_decay=raw_config['train']['main']['weight_decay'],
        steps=raw_config['train']['main']['steps'],
        batch_size=raw_config['train']['main']['batch_size'],
        check_val_every=raw_config['train']['main']['check_val_every'],
        sample_batch_size=raw_config['sample']['batch_size'],
        num_samples_to_generate=num_samples_to_generate,
        model_save_path=model_save_path,
        result_save_path=result_save_path,
        device=args.device,
        ckpt_path=ckpt_path,
        y_only=args.y_only
    )

    # Run or test
    if args.mode == 'train':
        with open(os.path.join(model_save_path, 'config.pkl'), 'wb') as f:
            pickle.dump(raw_config, f)
        trainer.run_loop()
    else:
        if args.report:
            trainer.report_test(args.num_runs)
        else:
            trainer.test()


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND-LINE INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training or sampling with TabDiff')

    # General
    parser.add_argument('--dataname',      type=str,   required=True)
    parser.add_argument('--mode',          choices=['train','test'], required=True)
    parser.add_argument('--gpu',           type=int,   default=0)
    parser.add_argument('--debug',         action='store_true')
    parser.add_argument('--no_wandb',      action='store_true')
    parser.add_argument('--exp_name',      type=str,   default=None)
    parser.add_argument('--deterministic', action='store_true')

    # Diffusion
    parser.add_argument('--y_only',               action='store_true')
    parser.add_argument('--non_learnable_schedule', action='store_true')

    # Sampling/Eval
    parser.add_argument('--num_samples_to_generate', type=int)
    parser.add_argument('--ckpt_path',               type=str)
    parser.add_argument('--report',                  action='store_true')
    parser.add_argument('--num_runs',                type=int, default=20)

    # Imputation
    parser.add_argument('--impute',             action='store_true')
    parser.add_argument('--trial_start',        type=int, default=0)
    parser.add_argument('--trial_size',         type=int, default=50)
    parser.add_argument('--resample_rounds',    type=int, default=1)
    parser.add_argument('--impute_condition',   type=str, default='x_t')
    parser.add_argument('--y_only_model_path',  type=str)
    parser.add_argument('--w_num',              type=float, default=0.6)
    parser.add_argument('--w_cat',              type=float, default=0.6)

    args = parser.parse_args()
    # Device
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    main(args)
