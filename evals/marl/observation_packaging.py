import numpy as np
from copy import deepcopy
from typing import Optional


def drones_obs_packaging(
        obs_history, 
        obs_map, 
        agent_list,
        min_obs=10, 
        noise=1.0
    ):

    obs_dict = {}

    for agent in agent_list:
        pos = obs_history[-1][agent][obs_map['self_pos']]
        allocentric_obs = []

        for i in range(min_obs):
            idx = min_obs - i
            if idx >= len(obs_history):
                idx = -1
            else:
                idx *= -1

            self_pos_ts = obs_history[idx][agent][obs_map['self_pos']] - pos
            target_obs_ts = obs_history[idx][agent][obs_map['target_pos']] + self_pos_ts #prev pos rel to current pos

            if i == min_obs - 1:
                obs_ts = target_obs_ts
                allocentric_obs = np.concatenate((np.array(allocentric_obs).flatten(),obs_ts))
            else:
                obs_ts = np.concatenate((self_pos_ts,target_obs_ts)).flatten()
                allocentric_obs.append(obs_ts)

        allocentric_obs = np.concatenate((allocentric_obs,obs_history[-1][agent][obs_map['target_goal']]))

        team_obs = obs_history[-1][agent][obs_map['team']]
        if len(obs_history) > 1:
            last_team_obs = obs_history[-2][agent][obs_map['team']]
            last_team_obs = last_team_obs + np.tile(obs_history[-2][agent][obs_map['self_pos']],len(last_team_obs)//3)
            last_team_obs = last_team_obs - np.tile(obs_history[-1][agent][obs_map['self_pos']],len(last_team_obs)//3)
        else:
            last_team_obs = deepcopy(team_obs)

        if noise is not None:
            noise = np.random.normal(0,noise,team_obs.shape)
        else:
            noise = np.zeros_like(team_obs)

        data_point = {}
        data_point['input'] = np.concatenate((allocentric_obs,last_team_obs + noise))
        data_point['label'] = np.concatenate((last_team_obs,team_obs))

        obs_dict[agent] = data_point

    return obs_dict, slice(len(last_team_obs),len(last_team_obs)+len(team_obs))

def predator_prey_obs_packaging(
        obs_history, 
        obs_map, 
        agent_list,
        min_obs=10, 
        noise: Optional[dict] = None
    ):

    obs_dict = {}

    for agent in agent_list:
        pos = obs_history[-1][agent][obs_map['self_pos']]
        allocentric_obs = []

        for i in range(min_obs):
            idx = min_obs - i
            if idx >= len(obs_history):
                idx = -1
            else:
                idx *= -1

            if noise is not None:
                target_noise = np.random.normal(0,noise['target'],(pos.shape))
            else:
                target_noise = np.zeros((2,))

            self_pos_ts = obs_history[idx][agent][obs_map['self_pos']] - pos
            target_obs_ts = obs_history[idx][agent][obs_map['target_pos']] + self_pos_ts + target_noise #prev pos rel to current pos

            if i == min_obs - 1:
                obs_ts = target_obs_ts
                allocentric_obs = np.concatenate((np.array(allocentric_obs).flatten(),obs_ts))
            else:
                obs_ts = np.concatenate((self_pos_ts,target_obs_ts)).flatten()
                allocentric_obs.append(obs_ts)

        allocentric_obs = np.concatenate((allocentric_obs,obs_history[-1][agent][obs_map['target_goal']]))

        team_obs = obs_history[-1][agent][obs_map['team']]
        if len(obs_history) > 1:
            last_team_obs = obs_history[-2][agent][obs_map['team']]
            last_team_obs = last_team_obs + np.tile(obs_history[-2][agent][obs_map['self_pos']],len(last_team_obs)//2)
            last_team_obs = last_team_obs - np.tile(obs_history[-1][agent][obs_map['self_pos']],len(last_team_obs)//2)
        else:
            last_team_obs = deepcopy(team_obs)

        if noise is not None:
            team_noise = np.random.normal(0,noise['team'],team_obs.shape)
        else:
            team_noise = np.zeros_like(team_obs)

        data_point = {}
        data_point['input'] = np.concatenate((allocentric_obs,last_team_obs + team_noise))
        data_point['label'] = np.concatenate((last_team_obs,team_obs))

        obs_dict[agent] = data_point

    return obs_dict, slice(len(last_team_obs),len(last_team_obs)+len(team_obs))