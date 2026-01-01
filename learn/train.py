import os
import sys
os.environ["OMP_NUM_THREADS"] = "1" #VERY IMPORTANT NP MATRIX INIVERSION WILL USE ALL CPUS IF NOT SET

import yaml
import torch
import shutil
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from envs.make_envs import make_usv_env
from learn.subproc_vec_env_no_daemon import SubprocVecEnvNoDaemon

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

def train(config_dir):
    cfg = load_config(config_dir)
    #edit values
    logdir = 'logs/' + cfg['logdir']

    if not os.path.exists(logdir):
        mkdir(logdir)

    #save configs
    with open(logdir+'/config.yaml', 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    # make parallel envs
    def env_fn(seed):
        env = make_usv_env(cfg)
        return env

    class EnvMaker:
        def __init__(self, seed):
            self.seed = seed

        def __call__(self):
            return env_fn(self.seed)


    def make_vec_env(nenvs, seed):
        envs = VecMonitor(
            SubprocVecEnvNoDaemon([EnvMaker(seed + i) for i in range(nenvs)])
        )
        return envs
    
    env = make_vec_env(cfg['alg']['nenvs'], cfg["seed"])

    #net_arch=dict(pi=cfg["alg"]["pi"], qf=cfg["alg"]["vf"])

    model = SAC(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=cfg['alg']['lr'],
            #target_entropy=0.1,
            tensorboard_log=logdir,
            )
    
    #checkpoint_callback = CheckpointCallback(
    #        save_freq=10000,
    #        save_path=logdir,
    #        name_prefix="midtrain_model",
    #        save_replay_buffer=False,
    #        save_vecnormalize=False,
    #        )

    model.learn(
            total_timesteps=float(cfg['alg']['timesteps']),
            progress_bar=True,
            #callback=checkpoint_callback,
            )


    # end
    model.save(os.path.join(logdir,"model"))

