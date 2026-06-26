import os
import cv2
import yaml
import copy
import shutil
import numpy as np
from gymnasium import spaces
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor

from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from envs.marl.make_env import make_marl_env
from learn.marl.train import make_ray_config, marl_policy_mapping_fn

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
        checkpoint_dir:str,
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
                checkpoint_dir,
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
        checkpoint_dir:str,
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
    min_obs = cfg['min_obs']
    cfg['min_obs'] = cfg['horizon'] * 10
    cfg['seed'] = seed 
    env = make_marl_env(
        cfg,
        seed=cfg['seed'],
        wrap='rllib',
        eval=True
    )
    obs_map = env.env.obs_map

    if checkpoint_dir is not None:
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        algo_config = make_ray_config(cfg)
        algo = algo_config.build_algo()
        algo.restore(checkpoint_dir)
    else:
        print('No model directory provided...')
        exit()

    policy_list = cfg['policy_list']
    policy_mapping_fn = marl_policy_mapping_fn

    for n in range(n_runs):

        save_dir = os.path.join(master_dir,str(n+dir_offset))
        mkdir(save_dir)

        obs, _ = env.reset()
        done = {"__all__": False}
        step = 0
        target_obs = {}
        obs_history = []

        episode_agent = np.random.choice(cfg['env']['learned_agent_list'])

        #for agent_name in cfg['env']['learned_agent_list']:
        target_obs[episode_agent] = []

        while not done["__all__"]:

            actions = {}

            for agent_id, agent_obs in obs.items():
                action = algo.compute_single_action(
                    agent_obs,
                    policy_id=policy_mapping_fn(agent_id, 0), 
                    explore=False
                )
                actions[agent_id] = action
            step += 1

            obs_history.append(obs[episode_agent][obs_map['team']])
            obs, rewards, terminations, truncations, infos = env.step(actions)

            done = {
                "__all__": terminations["__all__"] or truncations["__all__"]
            }


            if cfg['env']['scenario'] == 'predator_prey':
                rel_goal_pos = obs[episode_agent][obs_map['target_goal']]
                target_obs[episode_agent].append(obs[episode_agent][obs_map['target_pos']] - rel_goal_pos)

            ts_target_obs = []
            for i in range(min_obs):
                idx = min_obs - i
                if idx >= len(target_obs[episode_agent]):
                    ts_target_obs.append(target_obs[episode_agent][-1])
                else:
                    ts_target_obs.append(target_obs[episode_agent][-idx])
            ts_target_obs = np.array(ts_target_obs).flatten()

            if cfg['env']['scenario'] == 'predator_prey':
                team_obs = (obs_history[-1] - np.tile(rel_goal_pos,env.n_agents-1)) + np.random.uniform(-1.0,1.0,size=obs_history[-1].shape)

            full_obs = {}
            #for agent_name in cfg['env']['learned_agent_list']:
            full_obs[episode_agent] = {}
            full_obs[episode_agent]['target_true'] = np.concatenate((team_obs,ts_target_obs))
            full_obs[episode_agent]['team_true'] = np.concatenate((obs_history[-1],obs[episode_agent][obs_map['team']]-np.tile(rel_goal_pos,env.n_agents-1)))
            save_path = os.path.join(save_dir,'step_'+str(step)+'_'+episode_agent+'.npz')
            np.savez(save_path,**full_obs[episode_agent])

        obs, _ = env.reset()
    

