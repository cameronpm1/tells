import os
import sys
os.environ["OMP_NUM_THREADS"] = "1" #VERY IMPORTANT NP MATRIX INIVERSION WILL USE ALL CPUS IF NOT SET

import yaml
import torch
import shutil
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from envs.rl.make_envs import make_usv_env
from learn.rl.custom_callbacks import VideoCallback
from learn.rl.subproc_vec_env_no_daemon import SubprocVecEnvNoDaemon

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
    logdir = cfg['logdir']

    if not os.path.exists(logdir):
        mkdir(logdir)

    #save configs
    with open(logdir+'/config.yaml', 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    # make parallel envs
    def env_fn(seed):
        env = make_usv_env(cfg, seed=seed)
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
    
    env = make_vec_env(cfg['alg']['nenvs'], cfg['seed'])
    eval_env = make_vec_env(cfg['alg']['nenvs_eval'], cfg['seed']+cfg['alg']['nenvs'])

    if cfg["alg"]["type"] == "ppo":
        alg = PPO
        policy_kwargs = dict(net_arch=dict(pi=cfg["alg"]["pi"], vf=cfg["alg"]["vf"]))
        alg_kwargs = {}

    if cfg["alg"]["type"] == "sac":
        alg = SAC
        policy_kwargs = dict(net_arch=dict(pi=cfg["alg"]["pi"], qf=cfg["alg"]["vf"]))
        alg_kwargs = {'target_entropy' : 0.1}

    model = alg(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=cfg['alg']['lr'],
            tensorboard_log=logdir,
            policy_kwargs=policy_kwargs,
            **alg_kwargs,
            )
    
    video_callback = VideoCallback(
            eval_env, 
            best_model_save_path=logdir,
            log_path=logdir, 
            save_path=logdir + '/videos',
            eval_freq=int(100000/cfg['alg']['nenvs']),
            deterministic=True, 
            render=False)

    checkpoint_callback = CheckpointCallback(
            save_freq=int(100000/cfg['alg']['nenvs']),
            save_path=logdir,
            name_prefix="midtrain_model",
            save_replay_buffer=False,
            save_vecnormalize=False,
            )

    model.learn(
            total_timesteps=float(cfg['alg']['timesteps']),
            progress_bar=True,
            callback=[video_callback,checkpoint_callback],
            )


    # end
    model.save(os.path.join(logdir,"model"))

