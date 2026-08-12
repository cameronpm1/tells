import numpy as np
from copy import deepcopy
from typing import Optional

from controllers.football_control import get_ball_owner
from envs.marl.fire_env import DroneFireSim

# decomposed obs is captured before team markers are stamped in, so the fire
# window only ever contains GREEN/RED/BLACK/WHITE, never BLUE
FIRE_NUM_CLASSES = DroneFireSim.WHITE + 1

def _one_hot_fire_grid(grid: np.ndarray) -> np.ndarray:
    '''one-hots a fire grid into (num_classes, H, W), channels first'''
    return np.eye(FIRE_NUM_CLASSES, dtype=np.float32)[grid.astype(np.int64)].transpose(2, 0, 1)

def fire_obs_packaging(
        obs_history, 
        obs_map, 
        agent_list,
        min_obs=10, 
        noise: Optional[dict] = None
    ):

    obs_dict = {}

    for agent in agent_list:
        pos = obs_history[-1][agent]['self_pos']
        allocentric_obs = []

        pos_change = []

        for i in range(min_obs):
            idx = min_obs - i
            if idx >= len(obs_history):
                idx = -1
            else:
                idx *= -1

            self_pos_ts = obs_history[idx][agent]['self_pos'] - pos
            target_obs_ts = _one_hot_fire_grid(obs_history[idx][agent]['fire']).flatten() # #prev pos rel to current pos

            if i == min_obs - 1:
                obs_ts = target_obs_ts
                allocentric_obs = np.concatenate((np.array(allocentric_obs).flatten(),obs_ts))
                pos_change = np.array(pos_change).flatten()
            else:
                pos_change.append(self_pos_ts)
                obs_ts = target_obs_ts
                allocentric_obs.append(obs_ts)

        allocentric_obs = np.concatenate((allocentric_obs,pos_change))

        # team is a list of per-teammate relative positions (not a flat array like the other envs), so flatten it first
        team_obs = np.array(obs_history[-1][agent]['team']).flatten()
        if len(obs_history) > 1:
            last_team_obs = np.array(obs_history[-2][agent]['team']).flatten()
            last_team_obs = last_team_obs + np.tile(obs_history[-2][agent]['self_pos'],len(last_team_obs)//2)
            last_team_obs = last_team_obs - np.tile(obs_history[-1][agent]['self_pos'],len(last_team_obs)//2)
        else:
            last_team_obs = deepcopy(team_obs)

        if noise is not None:
            team_noise = np.random.normal(0, noise['team'], size=last_team_obs.shape)
        else:
            team_noise = np.zeros_like(last_team_obs)

        data_point = {}
        data_point['input'] = np.concatenate((allocentric_obs,last_team_obs + team_noise)) # add noise to input for this env
        data_point['label'] = np.concatenate((last_team_obs,team_obs)) # no noise for this env

        obs_dict[agent] = data_point

    return obs_dict, _transform_fire_pred

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

        #allocentric_obs = np.concatenate((allocentric_obs,obs_history[-1][agent][obs_map['target_goal']]))

        team_obs = obs_history[-1][agent][obs_map['team']]
        if len(obs_history) > 1:
            last_team_obs = obs_history[-2][agent][obs_map['team']]
            last_team_obs = last_team_obs + np.tile(obs_history[-2][agent][obs_map['self_pos']],len(last_team_obs)//3)
            last_team_obs = last_team_obs - np.tile(obs_history[-1][agent][obs_map['self_pos']],len(last_team_obs)//3)
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

    return obs_dict, _transform_drone_pred

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

def _transform_drone_pred(obs, obs_map, agent, prediction):

    obs[agent][obs_map['team']] = prediction[6:]

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

def _transform_fire_pred(obs, obs_map, agent, prediction):

    fire_map = np.array(obs[agent]['fire'], dtype=np.float32, copy=True)
    window_size = fire_map.shape[0]
    observation_range = (window_size - 1) // 2

    team_size = prediction.shape[0] // 2
    team_curr_pred = prediction[team_size:].reshape(-1, 2)

    for team_pos in team_curr_pred:
        rel_r = int(round(team_pos[0]))
        rel_c = int(round(team_pos[1]))

        if abs(rel_r) <= observation_range and abs(rel_c) <= observation_range:
            cross_r = rel_r + observation_range
            cross_c = rel_c + observation_range
            for rr, cc in [(cross_r, cross_c), (cross_r - 1, cross_c), (cross_r + 1, cross_c), (cross_r, cross_c - 1), (cross_r, cross_c + 1)]:
                if 0 <= rr < window_size and 0 <= cc < window_size:
                    fire_map[rr, cc] = DroneFireSim.BLUE

    obs[agent]['pred'] = fire_map[..., None]
    obs[agent]['team'] = prediction[team_size:]

    return obs