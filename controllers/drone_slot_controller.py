import numpy as np

def vec_to_vel_action(vec: np.ndarray, env, deadzone: float = 0.2) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < deadzone:
        return 0
    direction = vec / norm

    action_disc, action_vec = env.vec_to_action_mapper(direction)

    return action_disc


def _assign_slots(pursuer_positions: np.ndarray, slots: np.ndarray) -> list[np.ndarray]:
    # Same idea as the 2D controller: avoid forcing a drone across the formation
    # if another copy is already closer to that slot.
    remaining_slots = [slot.copy() for slot in slots]
    assigned_slots = []
    for pursuer_pos in pursuer_positions:
        slot_idx = int(
            np.argmin(
                [np.linalg.norm(pursuer_pos - slot) for slot in remaining_slots]
            )
        )
        assigned_slots.append(remaining_slots.pop(slot_idx))
    return assigned_slots


def compute_drone_slot_actions(env) -> dict[str, np.ndarray]:
    positions, _ = env._positions_and_velocities()
    pursuer_positions = positions[: env.num_pursuers]
    target_pos = positions[env.target_idx]
    goal_pos = env.GOAL_POS.astype(np.float32)
    target_goal_dist = float(np.linalg.norm(goal_pos - target_pos))
    pursuer_target_vecs = pursuer_positions - target_pos
    pursuer_target_vecs[:, 2] = 0.0
    avg_pursuer_dist = float(
        np.mean(np.linalg.norm(pursuer_target_vecs, axis=1))
    )

    slots = env.compute_slots(
        target_pos,
        goal_pos,
        target_goal_dist,
        avg_pursuer_target_dist=avg_pursuer_dist,
    )
    assign_slots = getattr(env, 'assign_slots', _assign_slots)
    assigned_slots = assign_slots(pursuer_positions, slots)

    actions = {}
    for idx, (agent, slot) in enumerate(zip(env.agents, assigned_slots)):
        actions[agent] = vec_to_vel_action(slot - pursuer_positions[idx],env)
    return actions
