import os
import sys
import yaml
import glob
import torch
import pprint
import shutil
from munch import munchify
from datetime import datetime
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from learn.belief.belief_model import BeliefModel

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)


def eval(
    config_dir: str,
    ckpt_dir: str,
):
    config_filepath = config_dir
    checkpoint_filepath = ckpt_dir
    checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[-1]

    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    log_dir = '_'.join([cfg.log_dir,
                        cfg.data_name,
                        cfg.model_name,
                        cfg.label])
    cfg['log_dir'] = log_dir

    model = BeliefModel(**cfg)

    ckpt = torch.load(checkpoint_filepath)
    model.load_state_dict(ckpt['state_dict'])

    model.eval()
    model.freeze()

    trainer = Trainer(deterministic=True,
                      default_root_dir=log_dir,
                      val_check_interval=1.0)

    trainer.test(model)
    model.test_save()

