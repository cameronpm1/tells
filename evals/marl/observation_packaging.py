import numpy as np
from copy import deepcopy
from typing import Optional

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
        ball_obs = obs_history[-1][agent][obs_map['ball_owner']]
        if len(obs_history) > 1:
            last_team_obs = obs_history[-2][agent][obs_map['team']]
            last_team_obs = last_team_obs + np.tile(obs_history[-2][agent][obs_map['self_pos']],len(last_team_obs)//2)
            last_team_obs = last_team_obs - np.tile(obs_history[-1][agent][obs_map['self_pos']],len(last_team_obs)//2)
            last_ball_obs = obs_history[-2][agent][obs_map['ball_owner']]
        else:
            last_team_obs = deepcopy(team_obs)
            last_ball_obs = ball_obs

        if noise is not None and agent in noise:
            team_noise = noise[agent]['team']
            team_noise_old = noise[agent]['team_old']
        else:
            team_noise = np.zeros_like(team_obs)
            team_noise_old = np.zeros_like(last_team_obs)

        data_point = {}
        data_point['input'] = np.concatenate((allocentric_obs,last_team_obs,last_ball_obs))
        data_point['label'] = np.concatenate((last_team_obs - team_noise_old,last_ball_obs,team_obs - team_noise,ball_obs))

        obs_dict[agent] = data_point

    team_start = len(last_team_obs) + len(last_ball_obs)
    ball_start = team_start + len(team_obs)
    obs_idxs = {
        'team': slice(team_start, team_start + len(team_obs)),
        'ball_owner': slice(ball_start, ball_start + len(ball_obs)),
    }

    return obs_dict, obs_idxs

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

    obs_idxs = {'team': slice(len(last_team_obs), len(last_team_obs) + len(team_obs))}

    return obs_dict, obs_idxs