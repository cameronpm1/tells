import time
import gymnasium
import numpy as np
from copy import deepcopy
from gymnasium import spaces
from collections import OrderedDict
from typing import Any, Dict, Type, Optional, Union
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class RLLibWrapper(MultiAgentEnv):
    '''
    wrapper for multiagent envs
    env should handle dict format, wrapper is
    for rllib MultiAgentEnv inheritance
    '''

    def __init__(
            self,
            env,
    ): 
        super().__init__()

        self.env = env
        self.agents = deepcopy(env.agents)
        self.possible_agents = deepcopy(env.agents)

        self.single_observation_spaces = {
            agent: self.env.observation_space(agent)
            for agent in self.agents
        }

        self.single_action_spaces = {
            agent: self.env.action_space(agent)
            for agent in self.agents
        }

        # disable deprecated paths
        self.observation_spaces = None
        self.action_spaces = None

        self.observation_space = None
        self.action_space = None

    def get_observation_space(self, agent):
        return self.single_observation_spaces[agent]

    def get_action_space(self, agent):
        return self.single_action_spaces[agent]
    
    def step(self, action_dict):

        terminated_all = False
        truncated_all = False

        obs,rew,terminated,truncated,_ = self.env.step(action_dict)

        terminated["__all__"] = all(terminated.values())
        truncated["__all__"] = all(truncated.values())
        #print(obs.keys(),rew.keys(),terminated.keys(),truncated.keys(),_.keys())

        return obs,dict(rew),terminated,truncated,{}
    
    def reset(self, **kwargs):
        obs,info = {},{}
        obs,info = self.env.reset(**kwargs)

        return obs,info
    
    def close(self):
        self.env.unwrapped.close()

    def render_rgb(self):
        return self.env.render_rgb()

    def set_difficulty(self, difficulty):
        self.env.set_difficulty(difficulty)