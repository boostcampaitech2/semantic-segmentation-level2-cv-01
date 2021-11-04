from copy import Error
import os
import re
import torch
import numpy as np
import random
import wandb

def make_save_dir(cfg, debug=False):
    """
    Make Directory to save checkpoints. This function has been added
    to avoid saving different experiments of same models in one directory.
    So you can prevent some of checkpoints from being changed.

    Args :
        cfg (str) : Config for path of directory to save checkpoints.

        debug (bool) : Debugging mode (default : False)

    Returns :
        saved_dir (str) : New directory path
    """

    if debug:
        return
        
    exp_name = set_exp_name(cfg, debug)

    if hasattr(cfg, 'saved_dir'):
        saved_dir = os.path.join(cfg.saved_dir, exp_name)
    else:
        saved_dir = os.path.join('/opt/ml/segmentation/saved', exp_name)

    while os.path.isdir(saved_dir):
        components = saved_dir.split('_')
        if len(components) > 1 and re.match("[0-9]{3}", components[-1]):
            suffix = components[-1]
            suffix = str(int(suffix) + 1).zfill(3)
            saved_dir = saved_dir[:-3] + suffix
        else:
            saved_dir += '_001'

    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)

    print(f"saved_dir has been created : {saved_dir}")

    return saved_dir

def set_random_seed(random_seed=21):
    """
    Set seed.

    Args:
        random_seed (int) : Seed to be set (default : 21)
    """
    random_seed = 21
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def set_exp_name(cfg, debug):
    """
    Set the name of an experiment.

    Args:
        cfg : Config for training.

        debug (bool): Debugging mode (default : False)

    Returns:
        exp_name (str): Name of the experiment.

    """
    pre_fix = ""
    exp_str = 'exp' + str(cfg.exp_num)
    if debug:
        pre_fix = 'debug_'
    if hasattr(cfg.wandb, 'run_name'):
        exp_name = pre_fix + '_'.join([cfg.wandb.run_name])
    else:
        exp_name = pre_fix + '_'.join([exp_str, cfg.model.type, cfg.model.args.encoder_name])

    return exp_name

def init_wandb(cfg, private=False, debug=False):
    """
    Initialize wandb run.

    Args:
        cfg : Config for tarining.

        private (bool): Private mode for logging data to private wandb project.
            (default : False)

        debug (bool): Debugging mode (default : False)
    """

    if private:
        entity = cfg.wandb.entity
    else:
        entity = cfg.wandb.team_entity
    project = cfg.wandb.project

    scheduler = None
    if hasattr(cfg, 'scheduler'):
        scheduler = cfg.scheduler

    wandb_cfg = dict(
        exp_num = cfg.exp_num,
        model = cfg.model,
        val_every = cfg.val_every,
        batch_size = cfg.batch_size,
        num_epochs = cfg.num_epochs,
        criterion = cfg.criterion,
        optimizer = cfg.optimizer,
        scheduler = scheduler,
        train_pipeline = [transform.type for transform in cfg.train_pipeline],
        test_pipeline = [transform.type for transform in cfg.test_pipeline]
    )

    wandb.init(project=project, entity=entity, config=wandb_cfg)
    wandb.run.name = set_exp_name(cfg, debug)
    wandb.run.save()