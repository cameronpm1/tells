import os
import sys
import time
import yaml
import hydra
import torch
import shutil
import random
import logging
import numpy as np
from torch import nn
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from ray.tune.logger import pretty_print
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.models import ModelCatalog
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.algorithms.sac.sac_torch_policy import SACTorchPolicy
from ray.rllib.utils.metrics import (
    NUM_ENV_STEPS_SAMPLED,
)

#from logger import getlogger
from util.util import mkdir, load_config
from envs.marl.rllib_wrapper import RLLibWrapper
from envs.marl.make_env import make_predator_prey_env

#logger = getlogger(__name__)

def train(config_path: str):

    cfg = load_config(config_path)
    logdir = cfg['logdir']
    if not os.path.exists(logdir):
        print('Save directory not found, creating path ...')
        #logger.info("Save directory not found, creating path ...")
        mkdir(logdir)
    else:
        print('Save directory found ...')
        #logger.info("Save directory found ...")
    print("current directory:", logdir)
    #save copy of config file
    OmegaConf.save(config=cfg, f=logdir+'/config.yaml')

    def logger_creator(config):
        return UnifiedLogger(config, logdir, loggers=None)

    algo_config = make_ray_config(cfg)

    algo_build = algo_config.build_algo(logger_creator=logger_creator)  

    #train 15,000 iterations
    for i in range(int(cfg['alg']['timesteps'])): 
        result = algo_build.train()
        if i % 500 == 0:# and i != 0:
            save_dir = logdir+'/checkpoint'+str(i)
            algo_build.save(checkpoint_dir=save_dir)
            print(pretty_print(result))

    save_dir = logdir+'/final'
    algo_build.save(checkpoint_dir=save_dir)


def make_ray_config(
    cfg: dict
):
    '''
    takes loaded config and returns ray config file for tarining rl

    input
    -----
    config:dict
        loaded config dictionary

    output
    ------
    ray_config class
    '''
    #make env function 
    def env_maker(config):

        seed = cfg['seed']

        if config is not None:
            seed += ((1000*config.worker_index + config.vector_index))   

        env = make_predator_prey_env(cfg,seed=int(seed),wrap=True)

        return env
        
    class policyTrainingSchedule():
        '''
        records how many batches have been used 
        '''
        def __init__(self,workers=1):
            self.workers=workers
            self.max_samples = 0

        def policy_training_schedule(self, pid, batch):
            '''
            determines when each policy should be trained
            '''
            #receives none when checking for target network update
            if batch is not None:
                self.max_samples = max(batch[pid]['unroll_id'])/len(batch.policy_batches)*self.workers
            
            return True
            
    policy_tracker = policyTrainingSchedule(workers=cfg['alg']['nenvs'])
    
    env_name = cfg['env']['scenario']
    register_env(env_name, env_maker) #register make env function
    #test_env for getting obs/action space
    test_env = env_maker(None)
    policy_list = cfg['policy_list']
    policy_mapping_fn = marl_policy_mapping_fn
    policy_training_fn = policy_tracker.policy_training_schedule

    batch = None

    if 'sac' in cfg['alg']['type']:

        algo = SACConfig() 
        policy_model_dict = {
            'post_fcnet_hiddens': cfg['alg']['pi'],
        }
        q_model_dict = {
            'post_fcnet_hiddens': cfg['alg']['vf'],
        }
        batch = cfg['alg']['batch']*len(cfg['policy_list'])

        policy_info = {}

        for label in policy_list:

            policy_info[label] = PolicySpec(
                            None, #policy_class
                            test_env.get_observation_space(label), #observation_space
                            test_env.get_action_space(label), #action_space
                            {'lr':cfg['alg']['lr'],
                                'policy_model_config':policy_model_dict,
                                'q_model_config':q_model_dict,
                            } #config (gamma,lr,etc.)
                        )

    #initialize RL training algorithm using rllib's multi_agent settings
    algo_config = (algo
            .environment(env=env_name,
                        #env_config={'num_agents':3},
                        )
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .framework("torch")
            .env_runners(num_env_runners=cfg['alg']['nenvs'], #20
                        num_envs_per_env_runner=cfg['alg']['cpu_envs'], #60
                        num_cpus_per_env_runner=1
                        )
            .resources(num_gpus=1)
            .multi_agent(policy_mapping_fn=policy_mapping_fn,
                            policies_to_train=policy_training_fn,
                            policies=policy_info)
            .training(gamma=cfg['alg']['gamma'], 
                        train_batch_size=batch,
                        training_intensity= None, #cfg['alg']['train_intensity'],
                        #target_entropy=cfg['alg']['target_ent'],
                        #grad_clip=10,
                        grad_clip_by='norm',
                        replay_buffer_config={
                            'type': 'MultiAgentReplayBuffer', #'EpisodeReplayBuffer', 
                            'capacity': 1000000, 
                            'replay_sequence_length': 1,
                            },
                        )
    )
    
    del test_env

    return algo_config

def marl_policy_mapping_fn(agent_id, episode, **kwargs):

    return 'agent0' #for single policy, multiple agents

    #return agent_id