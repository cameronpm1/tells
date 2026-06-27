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
    self_state = obs[obs_map["self"]]
    self_pos = self_state[3:6]

    target_block = obs[obs_map["target"]]
    n_boxes = max((len(target_block) // 3) - 1, 1)
    box_rel = target_block[: 3 * n_boxes].reshape(n_boxes, 3)
    adversary_rel = target_block[3 * n_boxes : 3 * (n_boxes + 1)]

    return self_pos, self_pos + box_rel, self_pos + adversary_rel


def _line_frame(box_positions: np.ndarray):
    xy = np.asarray(box_positions[:, 0:2], dtype=np.float32)
    center = np.mean(xy, axis=0)

    if len(xy) <= 1:
        axis = np.array([1.0, 0.0], dtype=np.float32)
    else:
        shifted = xy - center
        if float(np.linalg.norm(shifted)) < 1e-6:
            axis = np.array([1.0, 0.0], dtype=np.float32)
        else:
            _, _, vh = np.linalg.svd(shifted, full_matrices=False)
            axis = vh[0].astype(np.float32)

    major_axis = int(np.argmax(np.abs(axis)))
    if axis[major_axis] < 0.0:
        axis = -axis

    axis = axis / (np.linalg.norm(axis) + 1e-8)
    coords = (xy - center) @ axis
    return center, axis, coords


def _home_coords(box_coords: np.ndarray, num_protectors: int) -> np.ndarray:
    box_coords = np.sort(np.asarray(box_coords, dtype=np.float32))

    if num_protectors <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(box_coords) == 0:
        return np.zeros(num_protectors, dtype=np.float32)
    if num_protectors == 1:
        return np.array([float(np.mean(box_coords))], dtype=np.float32)
    if num_protectors > len(box_coords):
        return np.linspace(
            float(np.min(box_coords)),
            float(np.max(box_coords)),
            num_protectors,
            dtype=np.float32,
        )

    edges = np.rint(np.linspace(0, len(box_coords), num_protectors + 1)).astype(int)
    edges[0] = 0
    edges[-1] = len(box_coords)

    homes = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            homes.append(float(box_coords[min(start, len(box_coords) - 1)]))
        else:
            homes.append(float(np.mean(box_coords[start:stop])))
    return np.asarray(homes, dtype=np.float32)


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
    home: np.ndarray,
    box_coords: np.ndarray,
    adversary_pos: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    controller_cfg: dict,
) -> np.ndarray:
    if len(home) == 0:
        return home

    adversary_coord = float((adversary_pos[0:2] - center) @ axis)
    target_coord = float(
        np.clip(adversary_coord, float(np.min(box_coords)), float(np.max(box_coords)))
    )

    desired = home.copy()
    primary_idx = int(np.argmin(np.abs(home - target_coord)))
    response = float(controller_cfg.get("target_response", 1.0))
    desired[primary_idx] = home[primary_idx] + response * (target_coord - home[primary_idx])

    is_side_drone = primary_idx == 0 or primary_idx == len(home) - 1
    if is_side_drone or len(home) <= 2:
        return desired

    displacement = abs(float(desired[primary_idx] - home[primary_idx]))
    follow_threshold = float(controller_cfg.get("follow_threshold", 0.35))
    if displacement < follow_threshold:
        return desired

    base_gap = float(np.max(np.diff(home))) if len(home) > 1 else 0.0
    max_gap = float(controller_cfg.get("max_gap", base_gap * controller_cfg.get("max_gap_factor", 1.05)))
    return _limit_gaps_from_primary(desired, primary_idx, max_gap)


def _slots_to_positions(
    slot_coords: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    altitude: float,
) -> np.ndarray:
    slots = np.zeros((len(slot_coords), 3), dtype=np.float32)
    slots[:, 0:2] = center[None, :] + slot_coords[:, None] * axis[None, :]
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

    center, axis, box_coords = _line_frame(box_positions)
    home = _home_coords(box_coords, len(protector_positions))
    slot_coords = _desired_slot_coords(
        home=home,
        box_coords=box_coords,
        adversary_pos=adversary_pos,
        center=center,
        axis=axis,
        controller_cfg=controller_cfg,
    )

    altitude = float(controller_cfg.get("base_altitude", np.mean(box_positions[:, 2])))
    slots = _slots_to_positions(slot_coords, center, axis, altitude)

    protector_coords = (protector_positions[:, 0:2] - center) @ axis
    role_order = np.argsort(protector_coords)
    actions = np.zeros(len(protector_positions), dtype=np.int64)

    deadzone = float(controller_cfg.get("deadzone", 0.35))
    altitude_deadzone = float(controller_cfg.get("altitude_deadzone", 0.15))
    for role_idx, protector_idx in enumerate(role_order):
        move_vec = slots[role_idx] - protector_positions[protector_idx]
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
    if len(team_rel) > 0:
        other_positions = self_pos + team_rel.reshape(-1, 3)
        protector_positions = np.vstack([self_pos, other_positions])
    else:
        protector_positions = self_pos.reshape(1, 3)

    actions = _compute_actions_from_state(
        protector_positions=protector_positions,
        box_positions=box_positions,
        adversary_pos=adversary_pos,
        controller_cfg=controller_cfg,
    )
    return int(actions[0])


compute_drone_slot_actions = compute_drone_actions
