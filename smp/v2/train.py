import torch
import argparse

from init_api import make_save_dir, set_random_seed, init_wandb
from config_api import Config
from model_api import build_model
from data_api import build_loader
from train_api import train_model

def parse_args():
    parser = argparse.ArgumentParser(description='Train Segmentation Model')
    parser.add_argument('config', help='train config file path')

    parser.add_argument('--debug',
        action='store_true',
        help='whether to use small dataset for debugging')

    parser.add_argument('--deterministic',
        action='store_true',
        help='whether to set random seed for reproducibility')

    parser.add_argument('--private',
        action='store_true',
        help='whether to log to private wandb entity')
    
    args = parser.parse_args()

    return args

def main():
    # print init settings
    print("="*30 + '\n')
    print('pytorch version  : {}'.format(torch.__version__))
    print('GPU available    : {}'.format(torch.cuda.is_available()))
    print('GPU              : {}'.format(torch.cuda.get_device_name(0)))

    args = parse_args()
    config = args.config
    debug = args.debug
    deterministic = args.deterministic
    private = args.private

    print(f"Config          : {config}")
    print(f"Debugging Mode  : {debug}")
    print(f"Deterministic   : {deterministic}")
    print(f"Private wandb   : {private}")
    print('\n' + "="*30 + '\n')

    cfg = Config.fromfile(args.config)

    # create saved_dir
    saved_dir = make_save_dir(cfg, debug)

    # set random seed
    if deterministic:
        set_random_seed()

    # build model
    model = build_model(cfg.model)

    # build dataset
    train_loader = build_loader(cfg.data.train, debug=debug)
    valid_loader = build_loader(cfg.data.valid, debug=debug)

    # initialize wandb
    if hasattr(cfg, 'wandb'):
        init_wandb(cfg, private=private, debug=debug)

    # train
    train_model(
        model,
        train_loader,
        valid_loader,
        saved_dir,
        cfg,
        debug=debug)

if __name__ == '__main__':
    main()