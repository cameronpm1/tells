import time
import torch
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
            name,
            obs_packaging_func,
            eval: bool = False,
            belief_kwargs: Optional[dict] = None,
            dimension: int = 2,
    ): 
        super().__init__()

        self.env = env
        self.eval = eval
        self.name = name
        self.dim = dimension
        self.n_agents = env.n_agents
        self.belief_kwargs = belief_kwargs
        self.agents = deepcopy(env.agents)
        self.obs_packaging_func = obs_packaging_func
        self.last_raw_reward = None

        self.single_observation_spaces = {
            agent: self.env._observation_space(agent)
            for agent in self.agents
        }

        self.single_action_spaces = {
            agent: self.env._action_space(agent)
            for agent in self.agents
        }

        # disable deprecated paths
        self.observation_spaces = None
        self.action_spaces = None

        self.observation_space = None
        self.action_space = None

        if belief_kwargs is not None and belief_kwargs['on']:
            self.belief = True
            self.belief_model = belief_kwargs['model']
            self.belief_n = belief_kwargs['min_obs']
            self.obs_history = []
        else:
            self.belief = False

    def get_observation_space(self, agent):
        return self.single_observation_spaces[agent]

    def get_action_space(self, agent):
        return self.single_action_spaces[agent]
    
    def step(self, action_dict):

        terminated_all = False
        truncated_all = False

        obs,rew,terminated,truncated,infos = self.env.step(action_dict)
        rew = dict(rew)

        if self.eval:
            infos = {'target': obs['target']}
        else:
            infos = {}

        obs.pop("target", None)
        rew.pop('target', None)
        terminated.pop('target', None)
        truncated.pop('target', None)

        terminated["__all__"] = all(terminated.values())
        truncated["__all__"] = all(truncated.values())
        #print(obs.keys(),rew.keys(),terminated.keys(),truncated.keys(),_.keys())
        
        if self.belief:
            self.obs_history.append(obs)
        
        infos['__common__'] = {}
        infos['__common__']['raw_reward'] = sum(rew.values())

        self.last_raw_reward = sum(rew.values())

        #fill in observations
        if self.belief:
            converted_obs, obs_idxs = self.obs_packaging_func(self.obs_history, self.env.obs_map, self.agents, min_obs=self.belief_n, noise=None)

            predictions = {}
            errors = []
            for agent, agent_obs in obs.items():
                team_state = self.belief_model.model(converted_obs[agent]['input']).detach().cpu().numpy()[[obs_idxs]]
                error = self.permutation_invariant_error(team_state, agent_obs[self.env.obs_map['team']])
                self.obs_history[agent][self.env.obs_map['team']] = team_state
                errors.append(error)
                predictions[agent] = team_state
            avg_error = np.average(errors)
            infos['__common__']['belief_error'] = avg_error
            self.prediction_history.append(predictions)
        return obs,rew,terminated,truncated,infos
    
    def reset(self, **kwargs):
        obs,infos = {},{}
        obs,infos = self.env.reset(**kwargs)

        obs.pop("target", None)

        infos['__common__'] = {}
        infos['__common__']['raw_reward'] = 0.0
        self.last_raw_reward = 0.0

        if self.belief:
            self.obs_history = []
            self.prediction_history = []

            predictions = {}
            for agent in self.agents:
                predictions[agent] = obs[agent][self.env.obs_map['team']]

            self.obs_history.append(obs)
            self.prediction_history.append(predictions)

            infos['__common__']['belief_error'] = 0.0

        return obs,infos
    
    def close(self):
        self.env.unwrapped.close()

    def render_rgb(self):
        return self.env.render_rgb()

    def set_difficulty(self, difficulty):
        self.env.set_difficulty(difficulty)
        

    def permutation_invariant_error(self, pred, target):
        '''
        pred:   (N, 6)
        target: (N, 6)

        Returns:
            scalar, sum over batch of minimum assignment distances
        '''

        # Reshape to (N, 2, 3)
        pred = pred.reshape(-1, 2, 2)
        target = target.reshape(-1, 2, 2)

        # Direct assignment distances
        direct = (
            np.linalg.norm(pred[:, 0] - target[:, 0], axis=1) +
            np.linalg.norm(pred[:, 1] - target[:, 1], axis=1)
        )

        # Swapped assignment distances
        swapped = (
            np.linalg.norm(pred[:, 0] - target[:, 1], axis=1) +
            np.linalg.norm(pred[:, 1] - target[:, 0], axis=1)
        )

        # Take minimum per sample, then sum batch
        return np.minimum(direct, swapped).sum()







