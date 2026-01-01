import torch

from learn.train import train

if __name__ == "__main__":
    torch.set_num_threads(9)

    config_dir = 'confs/usv_configs/game1.yaml'
    train(config_dir)