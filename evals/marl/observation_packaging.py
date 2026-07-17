import numpy as np
from copy import deepcopy
from typing import Optional

from controllers.football_control import get_ball_owner

def football_obs_packaging(
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

            self_pos_ts = obs_history[idx][agent][obs_map['self_pos']] - pos
            target_obs_ts = obs_history[idx][agent][obs_map['target_pos']] + self_pos_ts # #prev pos rel to current pos

            if i == min_obs - 1:
                obs_ts = target_obs_ts
                allocentric_obs = np.concatenate((np.array(allocentric_obs).flatten(),obs_ts))
            else:
                obs_ts = np.concatenate((self_pos_ts,target_obs_ts)).flatten()
                allocentric_obs.append(obs_ts)

        team_obs = obs_history[-1][agent][obs_map['team']]
        ball_pos_obs = obs_history[-1][agent][obs_map['ball_pos']]
        if len(obs_history) > 1:
            last_team_obs = obs_history[-2][agent][obs_map['team']]
            last_team_obs = last_team_obs + np.tile(obs_history[-2][agent][obs_map['self_pos']],len(last_team_obs)//2)
            last_team_obs = last_team_obs - np.tile(obs_history[-1][agent][obs_map['self_pos']],len(last_team_obs)//2)
        else:
            last_team_obs = deepcopy(team_obs)

        if noise is not None and agent in noise:
            team_noise = noise[agent]['team']
            team_noise_old = noise[agent]['team_old']
            ball_noise = noise[agent].get('ball', np.zeros_like(ball_pos_obs))
        else:
            team_noise = np.zeros_like(team_obs)
            team_noise_old = np.zeros_like(last_team_obs)
            ball_noise = np.zeros_like(ball_pos_obs)

        data_point = {}
        data_point['input'] = np.concatenate((allocentric_obs,last_team_obs,ball_pos_obs))
        data_point['label'] = np.concatenate((last_team_obs - team_noise_old,team_obs - team_noise,ball_pos_obs - ball_noise))

        obs_dict[agent] = data_point

    return obs_dict, _transform_football_pred

def drones_obs_packaging(
        obs_history, 
        obs_map, 
        agent_list,
        min_obs=5, 
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

            self_pos_ts = obs_history[idx][agent][obs_map['self_pos']] - pos
            target_obs_ts = obs_history[idx][agent][obs_map['target_pos']] + self_pos_ts # #prev pos rel to current pos

            if i == min_obs - 1:
                obs_ts = target_obs_ts
                allocentric_obs = np.concatenate((np.array(allocentric_obs).flatten(),obs_ts))
            else:
                obs_ts = np.concatenate((self_pos_ts,target_obs_ts)).flatten()
                allocentric_obs.append(obs_ts)

        team_obs = obs_history[-1][agent][obs_map['team']]
        if len(obs_history) > 1:
            last_team_obs = obs_history[-2][agent][obs_map['team']]
            last_team_obs = last_team_obs + np.tile(obs_history[-2][agent][obs_map['self_pos']],len(last_team_obs)//2)
            last_team_obs = last_team_obs - np.tile(obs_history[-1][agent][obs_map['self_pos']],len(last_team_obs)//2)
        else:
            last_team_obs = deepcopy(team_obs)

        if noise is not None and agent in noise:
            team_noise = noise[agent]['team']
            team_noise_old = noise[agent]['team_old']
        else:
            team_noise = np.zeros_like(team_obs)
            team_noise_old = np.zeros_like(last_team_obs)

        data_point = {}
        data_point['input'] = np.concatenate((allocentric_obs,last_team_obs))
        data_point['label'] = np.concatenate((last_team_obs - team_noise_old,team_obs - team_noise))

        obs_dict[agent] = data_point

    obs_idxs = {'team': slice(len(last_team_obs), len(last_team_obs) + len(team_obs))}

    return obs_dict, obs_idxs

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

            self_pos_ts = obs_history[idx][agent][obs_map['self_pos']] - pos
            target_obs_ts = obs_history[idx][agent][obs_map['target_pos']] + self_pos_ts # #prev pos rel to current pos

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

        if noise is not None and agent in noise:
            team_noise = noise[agent]['team']
            team_noise_old = noise[agent]['team_old']
        else:
            team_noise = np.zeros_like(team_obs)
            team_noise_old = np.zeros_like(last_team_obs)

        data_point = {}
        data_point['input'] = np.concatenate((allocentric_obs,last_team_obs))
        data_point['label'] = np.concatenate((last_team_obs - team_noise_old,team_obs - team_noise))

        obs_dict[agent] = data_point

    return obs_dict, _transform_predator_prey_pred


def _transform_predator_prey_pred(obs, obs_map, agent, prediction):

    obs[agent][obs_map['team']] = prediction[4:]

    return obs

def _transform_football_pred(obs, obs_map, agent, prediction):

    team_size = obs_map['team'].stop - obs_map['team'].start

    team_curr_pred = prediction[team_size:2 * team_size]
    ball_pos_pred = prediction[2 * team_size:]

    obs[agent][obs_map['team']] = team_curr_pred
    obs[agent][obs_map['ball_pos']] = ball_pos_pred

    agent_idx = list(obs.keys()).index(agent)
    if obs[agent][obs_map['ball_owner']][agent_idx] != 1:
        ball_owner = get_ball_owner(team_curr_pred, ball_pos_pred, agent)
        obs[agent][obs_map['ball_owner']] = np.concatenate((ball_owner, np.zeros(1)))

    return obs