import numpy as np


def vec_to_action(vec: np.ndarray, deadzone: float = 0.05) -> int:
    if np.linalg.norm(vec) < deadzone:
        return 0
    if abs(vec[0]) > abs(vec[1]):
        return 2 if vec[0] > 0 else 1
    return 4 if vec[1] > 0 else 3

def _assign_slots(predators, slots):
    # Greedy matching is good enough here because we just need a stable expert to imitate,
    # not a globally optimal assignment solver every step.
    remaining_slots = [slot.copy() for slot in slots]
    assigned_slots = {}
    for predator in predators:
        slot_idx = int(
            np.argmin(
                [np.linalg.norm(predator.state.p_pos - slot) for slot in remaining_slots]
            )
        )
        assigned_slots[predator.name] = remaining_slots.pop(slot_idx)
    return assigned_slots


def compute_slot_actions(
    world,
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
    target = [agent for agent in world.agents if agent.adversary][0]
    predators = [agent for agent in world.agents if not agent.adversary]
    goal = world.landmarks[0]

    target_pos = target.state.p_pos.copy()
    goal_pos = goal.state.p_pos.copy()
    goal_vec = goal_pos - target_pos
    goal_dist = np.linalg.norm(goal_vec)
    forward = goal_vec / (goal_dist + 1e-8)
    lateral = np.array([-forward[1], forward[0]])
    avg_predator_dist = float(
        np.mean([np.linalg.norm(predator.state.p_pos - target_pos) for predator in predators])
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

    return {
        predator.name: vec_to_action(assigned_slots[predator.name] - predator.state.p_pos)
        for predator in predators
    }
