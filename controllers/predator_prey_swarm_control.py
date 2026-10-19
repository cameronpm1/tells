import numpy as np
from scipy.optimize import linear_sum_assignment

from controllers.predator_prey_control import vec_to_action, action_to_vec, adversary_controller


def _rotate(vec: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])

def _assign_slots(predators, slots):
    cost = np.linalg.norm(np.array(predators)[:, None] - np.array(slots)[None], axis=-1) ** 2
    _, slot_idx = linear_sum_assignment(cost)
    return [slots[i] for i in slot_idx]

def _arc_slots(center, direction, radius, half_angle, n, spacing, max_angle):
    # widen the arc to fit the swarm up to max_angle, then step outward a row once the arc is full
    half_angle = np.clip(n * spacing / (2 * radius), half_angle, max_angle)
    slots = []
    while len(slots) < n:
        k = min(n - len(slots), int(2 * half_angle * radius / spacing) + 1)
        angles = np.linspace(-half_angle, half_angle, k) if k > 1 else [0.0]
        slots.extend(center + radius * _rotate(direction, angle) for angle in angles)
        radius += spacing
    return slots

def compute_slot_actions(
    obs,
    obs_map,
    far_dist: float = 4.0,
    switch_dist: float = 2.2,
    funnel_radius: float = 3.4,
    funnel_angle: float = 0.65,
    gather_radius: float = 2.3,
    gather_angle: float = 0.65,
    push_radius: float = 2.0,
    push_angle: float = 0.95,
    goal_radius: float = 1.2,
    goal_angle: float = 1.57,
    spacing: float = 0.4,
    max_angle: float = 1.4,
    deadzone: float = 0.25,
) -> dict[str, int]:
    '''
    assumes recieves full marl obs dict, swarm obs does not contain individual teammate positions
    '''
    agent_names = [name for name in obs.keys() if 'agent' in name]
    predators = [obs[name][obs_map['self_pos']] for name in agent_names]
    target_pos = obs[agent_names[0]][obs_map['target_pos']] + predators[0]
    goal_pos = obs[agent_names[0]][obs_map['target_goal']] + predators[0]

    goal_vec = goal_pos - target_pos
    goal_dist = np.linalg.norm(goal_vec)
    backward = -goal_vec / (goal_dist + 1e-8)
    avg_predator_dist = float(
        np.mean([np.linalg.norm(predator_pos - target_pos) for predator_pos in predators])
    )

    if avg_predator_dist > far_dist:
        slots = _arc_slots(target_pos, backward, funnel_radius, funnel_angle, len(predators), spacing, max_angle)
    elif goal_dist > switch_dist:
        slots = _arc_slots(target_pos, backward, gather_radius, gather_angle, len(predators), spacing, max_angle)
    elif goal_dist > 1.0:
        slots = _arc_slots(target_pos, backward, push_radius, push_angle, len(predators), spacing, max_angle)
    else:
        slots = _arc_slots(goal_pos, backward, goal_radius, goal_angle, len(predators), spacing, max_angle)

    assigned_slots = _assign_slots(predators, slots)

    return {
        name: vec_to_action(slot - predator_pos, deadzone)
        for name, slot, predator_pos in zip(agent_names, assigned_slots, predators)
    }
