import os
import ray
import torch
import numpy as np

from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from evals.belief.eval import load_model
from envs.marl.make_env import make_marl_env
from envs.marl.rllib_wrapper import RLLibWrapper
from learn.marl.train import make_ray_config, marl_policy_mapping_fn
from util.util import mkdir, load_config, save_argb_video, save_rgb_gif 


def eval(
        config_dir:str,
        checkpoint_dir:str = None,
        n_runs: int = 1,
        belief_dir: str = None,
        belief_config_dir: str = None
    ):

    '''
    test policy and save gif of test run

    input
    -----
    config_dir:str
        config directory
    checkpoint_dir:str
        ray checkpoint directory
    n_runs:int
        number of runs to test for
    '''
    cfg = load_config(config_dir)
    belief_kwargs = {}

    env = make_marl_env(
        cfg,
        seed=cfg['seed'],
        wrap='pf',
        render_mode='rgb_array',
        belief_kwargs=belief_kwargs,
    )

    if checkpoint_dir is not None:
        algo_config = make_ray_config(cfg)
        algo = algo_config.build_algo()
        algo.restore(checkpoint_dir)
    else:
        print('No model directory provided...')
        exit()

    save_dir = os.path.join(checkpoint_dir,'videos')
    mkdir(save_dir)

    for i in range(n_runs):
        '''
        if belief:
            eval_single_episode_belief(
                env=env,
                cfg=cfg,
                algo=algo,
                belief_model=belief_model,
                save_dir=save_dir,
                idx=i,
            )
        else:
        '''
        eval_single_episode(
            env=env,
            cfg=cfg,
            algo=algo,
            save_dir=save_dir,
            idx=i
        )

def eval_single_episode(
        env,
        cfg:dict,
        algo,
        save_dir:str = '',
        idx:int=0,
    ):
    '''
    test sim for single episode

    inputs
    ------
    env: 
        environement
    cfg:dict
        config dictionary
    algo
        ray checkpoint loaded
    save_dir:str
        directory to save videos in
    idx:int
        index for saving video
    '''

    #register_env("predator_prey_eval", env_creator)

    episode_rewards = []
    episode_lengths = []
    images = []
    errors = []

    obs, infos = env.reset()
    if infos.get('__common__').get('belief_error') is not None:
        errors.append(infos['__common__']['belief_error'])
    done = {"__all__": False}

    total_reward = {agent: 0.0 for agent in obs.keys()}
    step_count = 0

    policy_list = cfg['policy_list']
    policy_mapping_fn = marl_policy_mapping_fn

    while not done["__all__"]:

        actions = {}

        for agent_id, agent_obs in obs.items():

            action = algo.compute_single_action(
                agent_obs,
                policy_id=policy_mapping_fn(agent_id, 0), 
                explore=False
            )
            actions[agent_id] = action

        obs, rewards, terminations, truncations, infos = env.step(actions)

        done = {
            "__all__": terminations["__all__"] or truncations["__all__"]
        }

        for agent_id, r in rewards.items():
            total_reward[agent_id] += r

        step_count += 1

        episode_rewards.append(sum(total_reward.values()))
        episode_lengths.append(step_count)
        if infos.get('__common__').get('belief_error') is not None:
            errors.append(infos['__common__']['belief_error'])

        images.append(env.render_rgb())

    save_file = str(os.path.join(save_dir,str(idx)+'.gif'))
    print('generating video in ' + save_file)
    save_rgb_gif(images,save_file)

    print("\n==== EVAL RESULTS ====")
    print(f"Reward: {episode_rewards[-1]}")
    print(f"Length: {episode_lengths[-1]}")
    if len(errors) > 0:
        print(f"Belief Error: {np.average(errors)}")
