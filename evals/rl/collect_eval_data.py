import os
import cv2
import yaml
import copy
import shutil
import numpy as np
from gymnasium import spaces
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor

from stable_baselines3 import SAC

from envs.make_envs import make_usv_env
from tells_environment_dynamics.test import *

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

def collect_data(
        config_dir:str,
        model_dir:str,
        master_dir:str,
        n_runs:int=1,
        seed:int=0,
        n_workers:int=1,
    ):
    '''
    test policy and save gif of test run

    input
    -----
    config_dir:str
        config directory
    model_dir:str
        policy directory
    master_dir:str
        directory to save gif
    n_runs:int
        number of test runs
    n_workers:int
        number of parallel workers to use for data collection
    '''
    mkdir(master_dir)

    cfg = load_config(config_dir)
    target_obs = cfg['min_obs']
    cfg['min_obs'] = cfg['horizon'] * 10
    env = make_usv_env(cfg,wrap=False,eval=True)

    run_segments = n_runs // n_workers
    run_segments_remainder = n_runs % n_workers

    args_list = []
    offset = 0

    for i in range(n_workers):
        if i == 0:
            n_runs = run_segments + run_segments_remainder
        else:
            n_runs = run_segments
        args_list.append(
            (
                config_dir,
                model_dir,
                master_dir,
                cfg['seed'] + i,
                n_runs,
                offset,
            )
        )
        offset += n_runs

    with Pool(n_workers) as pool:
        results = pool.starmap(data_worker, args_list)

    
def data_worker(
        config_dir:dict,
        model_dir:str,
        master_dir:str,
        seed:int,
        n_runs:int=1,
        dir_offset:int=0,
    ):
    '''
    make env and test policy for n_runs

    input
    -----
    config_dir:str
        config directory
    model_dir:str
        policy directory
    master_dir:str
        directory to save gif
    n_runs:int
        number of test runs
    dir_offset:int
        number of to offset save folders by (for parallel data collection)
    '''

    cfg = load_config(config_dir)
    target_obs = cfg['min_obs']
    cfg['min_obs'] = cfg['horizon'] * 10
    cfg['seed'] = seed 
    env = make_usv_env(cfg,wrap=False,eval=True)

    model = SAC.load(model_dir)

    for n in range(n_runs):

        obs, _ = env.reset()
        total_reward = 0
        save_dir = os.path.join(master_dir,str(n+dir_offset))
        mkdir(save_dir)

        for j in range(cfg['env']['max_episode_length']):

            action = np.zeros((6,))

            #must follow this order: ['chaser0','chaser1','chaser2','goals','target']
            #compute local observation for each boat (no noise in position for local pos est)
            offset = 0
            for i,b in enumerate(env.sim.boats.keys()):
                if 'chaser' not in b:
                    offset += 1
                else:
                    local_obs = []
                    for obs_b in env.sim.boats.keys():
                        if 'chaser' in obs_b:
                            if obs_b == b:
                                local_obs.extend(obs[obs_b+'_true'][-1])
                            else:
                                local_obs.extend(obs[obs_b])
                    for obs_g in obs['goals']:
                        local_obs.extend(obs_g)

                    for obs_t in obs['target'][-target_obs:]: #[0:target_obs]:
                        local_obs.extend(obs_t)

                    action[(i-offset)*2:(i-offset)*2+2] = model.predict(np.array(local_obs).flatten())[0][(i-offset)*2:(i-offset)*2+2]

            true_obs = {
                'target_true' : obs['target'],
            }
            for i in range(env.n_boats):
                true_obs['chaser'+str(i)+'_true'] = obs['chaser'+str(i)+'_true']
            true_obs['target_goal'] = obs['target_goal']


            save_path = os.path.join(save_dir,'step_'+str(j)+'.npz')
            np.savez(save_path,**true_obs)

            #action,_ = model.predict(obs)
            obs,rew,terminated,truncated,_ = env.step(action)
            total_reward += rew

            if terminated or truncated:
                print('Episode ended in ' + str(j) + ' timesteps')
                print('Total Reward: ' + str(total_reward))
                break
    


