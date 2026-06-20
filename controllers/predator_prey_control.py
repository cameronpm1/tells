import numpy as np


def vec_to_action(vec: np.ndarray, deadzone: float = 0.05) -> int:
    if np.linalg.norm(vec) < deadzone:
        return 0
    if abs(vec[0]) > abs(vec[1]):
        return 2 if vec[0] > 0 else 1
    return 4 if vec[1] > 0 else 3

def action_to_vec(action: int) -> int:
    if action == 1:
        return np.array([-1.0, 0.0])
    elif action == 2:
        return np.array([1.0, 0.0])
    elif action == 3:
        return np.array([0.0, -1.0])
    elif action == 4:
        return np.array([0.0, 1.0])
    else:
        return np.array([0.0, 0.0])

def _assign_slots(predators, slots):
    # Greedy matching is good enough here because we just need a stable expert to imitate,
    # not a globally optimal assignment solver every step.
    remaining_slots = [slot.copy() for slot in slots]
    assigned_slots = []
    for predator_pos in predators:
        slot_idx = int(
            np.argmin(
                [np.linalg.norm(predator_pos - slot) for slot in remaining_slots]
            )
        )
        assigned_slots.append(remaining_slots.pop(slot_idx))
    return assigned_slots

def compute_slot_actions(
    obs,
    obs_map,
    far_dist: float = 4.0,
    switch_dist: float = 2.2,
    funnel_backoff: float = 3.4,
    funnel_lateral: float = 2.6,
    gather_backoff: float = 2.3,
    gather_lateral: float = 1.7,
    push_radius: float = 2.1,
    push_offset: float = 1.1,
    flank_radius: float = 1.55,
    goal_backoff: float = 1.15,
    goal_lateral: float = 1.2,
) -> dict[str, int]:
    #if obs is dict, then recieved full marl obs
    if isinstance(obs,dict):
        target_pos = obs['agent0'][obs_map['target_pos']] + obs['agent0'][obs_map['self_pos']]
        predators = []
        for name in obs.keys():
            if 'agent' in name:
                predators.append(obs[name][obs_map['self_pos']])
        goal_pos = obs['agent0'][obs_map['target_goal']] + obs['agent0'][obs_map['self_pos']]
    else:
        target_pos = obs[obs_map['target_pos']] + obs[obs_map['self_pos']]
        predators = obs[obs_map['team']] + np.tile(obs[obs_map['self_pos']],2)
        predators = np.concatenate((obs[obs_map['self_pos']],predators)).reshape(-1,2)
        goal_pos = obs[obs_map['target_goal']] + obs[obs_map['self_pos']]

    goal_vec = goal_pos - target_pos
    goal_dist = np.linalg.norm(goal_vec)
    forward = goal_vec / (goal_dist + 1e-8)
    lateral = np.array([-forward[1], forward[0]])
    avg_predator_dist = float(
        np.mean([np.linalg.norm(predator_pos - target_pos) for predator_pos in predators])
    )

    if avg_predator_dist > far_dist:
        slots = [
            target_pos - forward * funnel_backoff,
            target_pos - forward * funnel_backoff + lateral * funnel_lateral,
            target_pos - forward * funnel_backoff - lateral * funnel_lateral,
        ]
    elif goal_dist > switch_dist:
        slots = [
            target_pos - forward * gather_backoff,
            target_pos - forward * gather_backoff + lateral * gather_lateral,
            target_pos - forward * gather_backoff - lateral * gather_lateral,
        ]
    elif goal_dist > 1.0:
        slots = [
            target_pos - forward * push_radius,
            target_pos - forward * push_offset + lateral * flank_radius,
            target_pos - forward * push_offset - lateral * flank_radius,
        ]
    else:
        slots = [
            goal_pos - forward * goal_backoff,
            goal_pos + lateral * goal_lateral,
            goal_pos - lateral * goal_lateral,
        ]

    assigned_slots = _assign_slots(predators, slots)

    if isinstance(obs,dict):
        obs.pop('target')
        actions = {}
        for i,name in enumerate(obs.keys()):
            actions[name] = vec_to_action(assigned_slots[i] - obs[name][obs_map['self_pos']])
    else:
        actions = vec_to_action(assigned_slots[-1] - obs[obs_map['self_pos']])

    return actions

def adversary_controller(
    obs: list[float],
    obs_map: dict,
    controller_cfg: dict
):

    force = np.zeros(2)

    agents = obs[obs_map['target_obs']].reshape(-1,2)

    for agent_pos in agents:

        diff = -agent_pos
        dist = np.linalg.norm(diff) + 1e-6
        local_gain = controller_cfg['prey_sensitivity']
        if dist < controller_cfg['prey_avoid_radius']:
            local_gain *= controller_cfg['prey_avoid_gain']
        force += local_gain * (diff / dist ** controller_cfg['force_exponent'])

    norm = np.linalg.norm(force)

    if norm > 0:
        force = force / norm

    # choose dominant direction
    if norm < controller_cfg['action_threshold']:
        return 0  # no-op

    if abs(force[0]) > abs(force[1]):
        return 2 if force[0] > 0 else 1   # right / left
    else:
        return 4 if force[1] > 0 else 3   # up / down