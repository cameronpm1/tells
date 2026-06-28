import numpy as np

from util.util import furthest_point

def vec_to_action(vec: np.ndarray, deadzone: float = 0.05) -> int:

    vec = np.asarray(vec, dtype=np.float32)

    if np.linalg.norm(vec) < deadzone:
        return 0  # stay still / release direction

    angle = np.arctan2(vec[0], vec[1])
    directions = [
        (np.array([-1.0,  0.0]), 1),  # left
        (np.array([-1.0, -1.0]), 2),  # top_left
        (np.array([ 0.0, -1.0]), 3),  # top
        (np.array([ 1.0, -1.0]), 4),  # top_right
        (np.array([ 1.0,  0.0]), 5),  # right
        (np.array([ 1.0,  1.0]), 6),  # bottom_right
        (np.array([ 0.0,  1.0]), 7),  # bottom
        (np.array([-1.0,  1.0]), 8),  # bottom_left
    ]

    closest_dir, action_idx = min(directions,key=lambda item: abs(np.arctan2(np.cross(vec,item[0]),np.dot(vec,item[0]))))

    return action_idx

def compute_target_aim(
    agent_obs,
    obs_map,
    target_proximity,
):
    pos = agent_obs[obs_map['self_pos']]
    target_pos = agent_obs[obs_map['target_pos']]
    target_vel = agent_obs[obs_map['target_vel']]
    #target_pos[0] *= -1
    #target_vel[1] *= -1
    team = np.concatenate((pos, agent_obs[obs_map['team']] + np.tile(pos,4))).reshape(-1,2)

    team_rel_target = []
    for j,team_pos in enumerate(team):
        team_rel_target.append((team_pos - (target_pos + pos),j))
    target_aim_vec, target_aim_idx = min(team_rel_target,key=lambda item: np.linalg.norm(item[0]))
    if np.linalg.norm(target_aim_vec) > target_proximity:
        target_aim_vec, target_aim_idx = min(team_rel_target,key=lambda item: abs(np.arctan2(np.cross(target_vel,item[0]),np.dot(target_vel,item[0]))))

    return target_aim_vec, target_aim_idx

def compute_rondo_actions(
    obs,
    obs_map,
    anchor_max_dev: list[float] = 0.2,
    closest_angle_2_target: float = 0.6,
    boundary_goal_proximity: float = 0.1,
    goal_point_angle_proximity: float = 0.2,
    target_proximity: float = 0.05,
) -> dict[str, int]:

    if isinstance(obs,dict):
        agent_names = [name for name in obs.keys() if 'agent' in name]
        obs_list = [obs[agent_name] for agent_name in agent_names]
    else:
        obs_list = [obs]

    actions = []

    for i,agent_obs in enumerate(obs_list):
        pos = agent_obs[obs_map['self_pos']]
        vel = agent_obs[obs_map['self_vel']]

        anchor = agent_obs[obs_map['target_ball']][0:2]

        #if i == 0:
        #    target_aim_vec, target_aim_idx = compute_target_aim(agent_obs, obs_map, target_proximity)
        if int(agent_obs[obs_map['target_ball']][-1]):
            ball_owner = i
        else:
            ball_owner = agent_obs[obs_map['team']][-1]

        team_rel = agent_obs[obs_map['team']][0:-1].reshape(-1,2)
        target_rel = agent_obs[obs_map['target_pos']]

        action = None

        if ball_owner == i: #int(agent_obs[obs_map['target_ball']][-1]) or target_aim_idx == i:
            #agent has ball or target is moving towards this player, try to optimize space
            team_angles_from_target = [abs(np.arctan2(np.cross(target_rel,team_i_rel),np.dot(target_rel,team_i_rel))) for team_i_rel in team_rel]
            idx, safest_member = max(enumerate(team_rel),key=lambda item: team_angles_from_target[item[0]] * int(team_angles_from_target[item[0]] > closest_angle_2_target))
            desired_vec = team_rel[idx]
            angle_to_goal = abs(np.arctan2(np.cross(vel,desired_vec),np.dot(vel,desired_vec)))

            if angle_to_goal < goal_point_angle_proximity and np.linalg.norm(vel) > 0.001:
                action = 9
        elif (i - ball_owner) % 5 in (1, 5 - 1):
            #agent is neighboring an agent with the ball and should expand the circle
            desired_point = anchor * 1.5 #expand the circle
            desired_vec = desired_point - pos
        elif np.linalg.norm(anchor-pos) > anchor_max_dev:
            desired_vec = anchor - pos
        else:
            desired_vec = anchor - pos

        if action is None:
            if np.linalg.norm(desired_vec) > 0.002:
                action = vec_to_action(desired_vec)
            else:
                action = 0

        actions.append(action)

    if isinstance(obs,dict):
        actions_dict = {}
        for i,name in enumerate(agent_names):
            actions_dict[name] = actions[i]
        return actions_dict
    else:
        return actions[0]



def adversary_controller(
    obs: list[float],
    obs_map: dict,
    controller_cfg: dict
):
    '''
    assumes recieves obs['target'] from full state obs
    '''

    force = np.zeros(2)

    rel_ball_pos = obs[obs_map['target_ball']][0:2]

    if np.linalg.norm(rel_ball_pos) < controller_cfg['dead_zone']:
        vec = 0.0
    else:
        vec = rel_ball_pos / np.linalg.norm(rel_ball_pos)
    
    return vec_to_action(vec, deadzone=0.0001)