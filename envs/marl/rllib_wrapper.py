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
        self.possible_agents = deepcopy(env.agents)
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
            self.belief_n = 10
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
            partial_obs = {}
            for agent in self.agents:
                partial_obs[agent+'_partial'] = obs[agent][self.env.obs_map['target']]
            self.obs_history.append(partial_obs)
        
        infos['__common__'] = {}
        infos['__common__']['raw_reward'] = sum(rew.values())

        self.last_raw_reward = sum(rew.values())

        #fill in observations
        if self.belief:
            converted_obs = self.convert_predator_prey_obs(self.obs_history, self.prediction_history)

            predictions = {}
            errors = []
            for agent_id, agent_obs in obs.items():
                if 'partial' in agent_id:
                    continue
                team_state = self.belief_model.model(converted_obs[agent_id+'_partial'])
                predictions[agent_id] = team_state.detach().cpu().numpy()[-self.dim*(self.n_agents-1):]
                team_state = team_state.detach().cpu().numpy()[-self.dim*(self.n_agents-1):] + np.tile(partial_obs[agent_id+'_partial'][2:],2)
                error = self.permutation_invariant_error(team_state, obs[agent_id][self.env.obs_map['team']])
                agent_obs[self.env.obs_map['team']] = team_state
                predictions[agent_id] = team_state
                errors.append(error)
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
            partial_obs = {}
            predictions = {}
            for agent in self.agents:
                partial_obs[agent+'_partial'] = obs[agent][self.env.obs_map['target']]
                predictions[agent] = obs[agent][self.env.obs_map['team']]
            self.prediction_history.append(predictions)

            self.obs_history.append(partial_obs)
            converted_obs = self.convert_predator_prey_obs(self.obs_history, self.prediction_history)

            predictions = {}
            errors = []
            for agent_id, agent_obs in obs.items():
                if 'partial' in agent_id:
                    continue
                team_state = self.belief_model.model(converted_obs[agent_id+'_partial'])
                team_state = team_state.detach().cpu().numpy()[-self.dim*(self.n_agents-1):] + np.tile(partial_obs[agent_id+'_partial'][-self.dim:],2)
                predictions[agent_id] = team_state
                error = self.permutation_invariant_error(team_state, obs[agent_id][self.env.obs_map['team']])
                agent_obs[self.env.obs_map['team']] = team_state
                errors.append(error)
            avg_error = np.average(errors)
            infos['__common__']['belief_error'] = avg_error
            self.prediction_history.append(predictions)


        return obs,infos
    
    def close(self):
        self.env.unwrapped.close()

    def render_rgb(self):
        return self.env.render_rgb()

    def set_difficulty(self, difficulty):
        self.env.set_difficulty(difficulty)

    def convert_predator_prey_obs(self,obs,predictions):
        '''
        convert observation history into timeseries of relateive advesrary and goal observations
        '''

        converted_obs = {}

        for agent in obs[0].keys():
            if 'partial' in agent:
                agent_obs = []
                offset = np.tile(obs[-1][agent][-self.dim:],2)
                for i in range(self.belief_n):
                    if i >= len(obs):
                        agent_obs.append(obs[0][agent] - offset)
                    else:
                        agent_obs.append(obs[-(i+1)][agent] - offset)
                agent_obs.reverse()
                agent_obs = np.array(agent_obs).flatten()
                
                if len(predictions) == 1:
                    team_obs = predictions[-1][agent.split('_')[0]] - offset
                else:
                    team_obs = predictions[-1][agent.split('_')[0]] - offset

                converted_obs[agent] = torch.from_numpy(np.concatenate((team_obs,agent_obs))).to(torch.float32)


        return converted_obs

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







