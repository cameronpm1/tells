import numpy as np

# simple and hopefully actually working controller yay

def vec_to_vel_action(vec: np.ndarray, deadzone: float = 0.04) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < deadzone:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    direction = vec / norm
    return np.array(
        [
            np.clip(direction[0], -1.0, 1.0),
            np.clip(direction[1], -1.0, 1.0),
            np.clip(direction[2], -1.0, 1.0),
        ],
        dtype=np.float32,
    )


def compute_drone_slot_actions(env) -> dict[str, np.ndarray]:
    positions, _ = env._positions_and_velocities()
    pursuer_positions = positions[: env.num_pursuers]
    target_pos = positions[env.target_idx]
    goal_pos = env.GOAL_POS.astype(np.float32)
    target_goal_dist = float(np.linalg.norm(goal_pos - target_pos))

    slots = env.compute_slots(target_pos, goal_pos, target_goal_dist)

    actions = {}
    for idx, agent in enumerate(env.agents):
        # Keep slot ownership stable so BC teaches the shared policy distinct roles.
        role_slot = slots[idx % len(slots)]
        actions[agent] = vec_to_vel_action(role_slot - pursuer_positions[idx])
    return actions
