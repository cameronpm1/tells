from __future__ import annotations

import numpy as np


_ACTION_TO_VEC = {
    0: np.array([0.0, 0.0, 0.0]),
    1: np.array([1.0, 0.0, 0.0]),
    2: np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0.0]),
    3: np.array([0.0, 1.0, 0.0]),
    4: np.array([-1 / np.sqrt(2), 1 / np.sqrt(2), 0.0]),
    5: np.array([-1.0, 0.0, 0.0]),
    6: np.array([-1 / np.sqrt(2), -1 / np.sqrt(2), 0.0]),
    7: np.array([0.0, -1.0, 0.0]),
    8: np.array([1 / np.sqrt(2), -1 / np.sqrt(2), 0.0]),
    9: np.array([0.0, 0.0, 1.0]),
    10: np.array([0.0, 0.0, -1.0]),
}


def action_to_vec(action: int) -> np.ndarray:
    return _ACTION_TO_VEC.get(int(action), _ACTION_TO_VEC[0]).copy()


def vec_to_action(
    vec: np.ndarray,
    deadzone: float = 0.35,
    altitude_deadzone: float = 0.15,
) -> int:
    vec = np.asarray(vec, dtype=np.float32)
    horizontal = vec[0:2]
    horizontal_norm = float(np.linalg.norm(horizontal))

    if horizontal_norm < deadzone and abs(float(vec[2])) < altitude_deadzone:
        return 0

    if abs(float(vec[2])) >= altitude_deadzone and abs(float(vec[2])) > horizontal_norm:
        return 9 if vec[2] > 0.0 else 10

    if horizontal_norm < deadzone:
        return 0

    unit = horizontal / horizontal_norm
    action_dirs = np.array(
        [
            [1.0, 0.0],
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
            [0.0, 1.0],
            [-1 / np.sqrt(2), 1 / np.sqrt(2)],
            [-1.0, 0.0],
            [-1 / np.sqrt(2), -1 / np.sqrt(2)],
            [0.0, -1.0],
            [1 / np.sqrt(2), -1 / np.sqrt(2)],
        ],
        dtype=np.float32,
    )
    return int(np.argmax(action_dirs @ unit) + 1)


def _agent_names(obs: dict) -> list[str]:
    return sorted(name for name in obs.keys() if name != "target")


def _parse_local_obs(obs: np.ndarray, obs_map: dict):
    obs = np.asarray(obs, dtype=np.float32)
    self_state = obs[obs_map['self']]
    self_pos = self_state[3:6]

    box_rel = obs[obs_map['target_goal']].reshape(-1, 3)
    adversary_rel = obs[obs_map['target_pos']]
    n_boxes = len(box_rel)

    return self_pos, self_pos + box_rel, self_pos + adversary_rel


def _limit_gaps_from_primary(
    desired: np.ndarray,
    primary_idx: int,
    max_gap: float,
) -> np.ndarray:
    desired = desired.copy()

    for idx in range(primary_idx + 1, len(desired)):
        desired[idx] = min(desired[idx], desired[idx - 1] + max_gap)

    for idx in range(primary_idx - 1, -1, -1):
        desired[idx] = max(desired[idx], desired[idx + 1] - max_gap)
    return desired


def _desired_slot_coords(
    box_positions: np.ndarray,
    adversary_pos: np.ndarray,
    controller_cfg: dict,
) -> np.ndarray:
    """
    Place 3 slots on a 120-degree arc (1/3 of a circle), each at distance
    `slot_spacing`, centered on an anchor point that sits between the
    adversary and its closest box -- forming a barrier between the two.

    The anchor point lies along the adversary-to-closest-box direction, at
    distance `target_spacing` from the adversary (or at the box itself if
    the box is closer than `target_spacing`). All 3 slots share that same
    forward distance from the adversary -- `slot_spacing` only spreads them
    sideways along the arc, it never pushes a slot further forward. Finally,
    each slot is clipped so it is never further than `drone_leash` from the
    center box.
    """
    slot_spacing = float(controller_cfg.get("slot_spacing", 1.0))
    target_spacing = float(controller_cfg.get("target_spacing", 1.0))
    drone_leash = float(controller_cfg.get("drone_leash", np.inf))

    adversary_xy = adversary_pos[0:2]
    box_xy = box_positions[:, 0:2]
    center_box_xy = box_xy[len(box_xy) // 2]

    dists_to_boxes = np.linalg.norm(box_xy - adversary_xy, axis=1)
    closest_box_xy = box_xy[int(np.argmin(dists_to_boxes))]

    direction = closest_box_xy - adversary_xy
    dist_to_closest_box = float(np.linalg.norm(direction))
    direction = direction / dist_to_closest_box

    forward = min(target_spacing, dist_to_closest_box)
    perp = np.array([-direction[1], direction[0]])

    slot_xy = []
    for offset in (-np.pi / 3, 0.0, np.pi / 3):
        sideways = slot_spacing * np.sin(offset)
        slot = adversary_xy + forward * direction + sideways * perp

        rel_to_center = slot - center_box_xy
        dist_to_center = float(np.linalg.norm(rel_to_center))
        if dist_to_center > drone_leash:
            slot = center_box_xy + rel_to_center / dist_to_center * drone_leash

        slot_xy.append(slot)

    return np.array(slot_xy), perp

def _slots_to_positions(
    slot_coords: np.ndarray,
    altitude: float,
) -> np.ndarray:
    slots = np.zeros((len(slot_coords), 3), dtype=np.float32)
    slots[:, 0:2] = slot_coords
    slots[:, 2] = float(altitude)
    return slots


def _compute_actions_from_state(
    protector_positions: np.ndarray,
    box_positions: np.ndarray,
    adversary_pos: np.ndarray,
    controller_cfg: dict | None = None,
) -> np.ndarray:
    controller_cfg = controller_cfg or {}
    protector_positions = np.asarray(protector_positions, dtype=np.float32).reshape(-1, 3)
    box_positions = np.asarray(box_positions, dtype=np.float32).reshape(-1, 3)
    adversary_pos = np.asarray(adversary_pos, dtype=np.float32)

    slot_coords, perp = _desired_slot_coords(
        box_positions=box_positions,
        adversary_pos=adversary_pos,
        controller_cfg=controller_cfg,
    )

    altitude = float(controller_cfg.get("base_altitude", np.mean(box_positions[:, 2])))
    slots = _slots_to_positions(slot_coords, altitude)

    protector_sideways = (protector_positions[:, 0:2] - adversary_pos[0:2]) @ perp
    role_order = np.argsort(protector_sideways)
    actions = np.zeros(len(protector_positions), dtype=np.int64)

    deadzone = float(controller_cfg.get("deadzone", 0.35))
    altitude_deadzone = float(controller_cfg.get("altitude_deadzone", 0.15))
    min_altitude = float(controller_cfg.get("min_altitude", 0.25))
    max_altitude = float(controller_cfg.get("max_altitude", 1.5))
    for role_idx, protector_idx in enumerate(role_order):
        move_vec = slots[role_idx] - protector_positions[protector_idx]
        altitude = float(protector_positions[protector_idx, 2])
        if altitude < min_altitude:
            actions[protector_idx] = 9
        elif altitude > max_altitude:
            actions[protector_idx] = 10
        else:
            actions[protector_idx] = vec_to_action(move_vec, deadzone, altitude_deadzone)

    return actions


def compute_drone_actions(
    obs,
    obs_map: dict,
    controller_cfg: dict | None = None,
):
    """
    Expert protector controller for the drone box-defense environment.

    The protectors keep fixed coverage slots on the goal-box line. A side
    protector can move inward toward adversary pressure while the other drones
    hold. When a middle protector moves far enough to open a larger gap, the
    neighbor on the widening side follows just enough to keep the gap bounded.
    """
    if isinstance(obs, dict):
        agent_names = _agent_names(obs)
        if not agent_names:
            return {}

        _, box_positions, adversary_pos = _parse_local_obs(obs[agent_names[0]], obs_map)
        protector_positions = np.asarray(
            [_parse_local_obs(obs[name], obs_map)[0] for name in agent_names],
            dtype=np.float32,
        )
        actions = _compute_actions_from_state(
            protector_positions=protector_positions,
            box_positions=box_positions,
            adversary_pos=adversary_pos,
            controller_cfg=controller_cfg,
        )
        return {name: int(actions[idx]) for idx, name in enumerate(agent_names)}

    self_pos, box_positions, adversary_pos = _parse_local_obs(obs, obs_map)
    team_rel = np.asarray(obs, dtype=np.float32)[obs_map["team"]]
    other_positions = self_pos + team_rel.reshape(-1, 3)
    protector_positions = np.vstack([self_pos, other_positions])

    actions = _compute_actions_from_state(
        protector_positions=protector_positions,
        box_positions=box_positions,
        adversary_pos=adversary_pos,
        controller_cfg=controller_cfg,
    )
    
    return int(actions[0])


def drone_controller(obs, obs_map: dict, controller_cfg: dict):
    """Predator-prey style entry point: obs, obs_map, controller constants."""
    drone_actions = compute_drone_actions(obs, obs_map, controller_cfg)
    return drone_actions


controller = drone_controller
compute_drone_slot_actions = compute_drone_actions
