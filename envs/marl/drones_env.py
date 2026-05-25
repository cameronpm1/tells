"""
Predator-prey drone environment for gym-pybullet-drones.

Learner controls N pursuer drones.
The final drone is a scripted target drone.
Reward is shared and shaped around 3D herding:
    move the target toward the goal, keep a useful formation, then hold it there.

Recommended action type:
    ActionType.VEL

Learner actions are 3D directions. The env appends a fixed speed fraction so
the pursuers stay about twice as fast as the scripted target.
"""

import numpy as np
import pybullet as p
import sys
from gymnasium import spaces
from pathlib import Path
from itertools import permutations as _permutations

_PYBULLET_DRONES_PATH = Path(__file__).resolve().parents[2] / "external" / "pybullet-drones"
if _PYBULLET_DRONES_PATH.exists() and str(_PYBULLET_DRONES_PATH) not in sys.path:
    sys.path.insert(0, str(_PYBULLET_DRONES_PATH))

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class DronesEnv(BaseRLAviary):
    def __init__(
        self,
        agent_list: list[str],
        learned_agent_list: list[str],
        grid_size: float = 3.0,
        goal_pos: np.ndarray | None = None,
        capture_radius: float = 0.5,
        base_altitude: float = 1.0,
        speed_ratio: float = 0.4,
        pursuer_speed_fraction: float = 1.5,
        episode_len_sec: int = 25,
        target_force_threshold: float = 1.5,
        pursuer_spawn_min_radius: float = 1.0,
        pursuer_spawn_max_radius: float = 2.2,
        pursuer_same_spawn: bool = True,
        pursuer_same_spawn_jitter: float = 0.05,
        target_spawn_radius: float = 1.6,
        spawn_altitude_jitter: float = 0.05,
        goal_noise: float = 0.1,
        global_observations: bool = False,
        role_conditioned_slots: bool = False,
        reward_kwargs: dict | None = None,
        controller_kwargs: dict | None = None,
        seed: int | None = None,
        drone_model: DroneModel = DroneModel.CF2X,
        neighbourhood_radius: float = np.inf,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui: bool = False,
        record: bool = False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.VEL,
    ):
        if act != ActionType.VEL:
            raise ValueError("PredatorPreyAviary currently assumes ActionType.VEL.")
        self.agents = learned_agent_list
        self.all_agents = agent_list
        self.num_pursuers = len(learned_agent_list)
        self.target_idx = self.num_pursuers
        self.grid_size = float(grid_size)
        self.capture_radius = float(capture_radius)
        self.base_altitude = float(base_altitude)
        self.speed_ratio = float(speed_ratio)
        self.pursuer_speed_fraction = float(pursuer_speed_fraction)
        self.EPISODE_LEN_SEC = float(episode_len_sec)
        self.target_force_threshold = float(target_force_threshold)
        self.pursuer_spawn_min_radius = float(pursuer_spawn_min_radius)
        self.pursuer_spawn_max_radius = float(pursuer_spawn_max_radius)
        self.pursuer_same_spawn = bool(pursuer_same_spawn)
        self.pursuer_same_spawn_jitter = float(pursuer_same_spawn_jitter)
        self.target_spawn_radius = float(target_spawn_radius)
        self.spawn_altitude_jitter = float(spawn_altitude_jitter)
        self.goal_noise = float(goal_noise)
        self.global_observations = bool(global_observations)
        self.role_conditioned_slots = bool(role_conditioned_slots)

        self.GOAL_POS = (
            np.array([0.0, 0.0, self.base_altitude], dtype=np.float32)
            if goal_pos is None
            else np.asarray(goal_pos, dtype=np.float32).copy()
        )
        self._fixed_goal_pos = goal_pos is not None

        self.reward_cfg = {
            'distance_scale': 2.0,
            'chase_scale': 0.25,
            'progress_scale': 18.0,
            'approach_scale': 3.0,
            'containment_scale': 2.0,
            'coverage_scale': 2.0,
            'slot_scale': 18.0,
            'hold_scale': 20.0,
            'success_bonus': 750.0,
            'oob_penalty': 3000.0,
            'uncontrolled_goal_scale': 12.0,
            'touch_penalty_scale': 4.0,
            'altitude_scale': 1.0,
            'step_cost': 0.05,
            'goal_focus_temp': 0.75,
            'surround_radius': 1.35,
            'ideal_radius': 1.0,
            'radius_tolerance': 0.45,
            'hold_goal_radius': 0.45,
            'hold_coverage_min': 0.30,
            'hold_close_fraction': 0.34,
            'hold_ring_min': 0.05,
            'hold_slot_min': 0.05,
            'goal_center_radius': 0.60,
            'goal_center_scale': 14.0,
            'goal_lock_radius': 0.25,
            'goal_lock_scale': 32.0,
            'success_hold_steps': 8,
            'slot_far_distance': 1.75,
            'slot_funnel_backoff': 1.55,
            'slot_funnel_lateral': 1.10,
            'slot_gather_distance': 1.20,
            'slot_gather_backoff': 1.25,
            'slot_gather_lateral': 0.95,
            'slot_switch_distance': 0.9,
            'slot_push_radius': 1.05,
            'slot_push_offset': 0.65,
            'slot_flank_radius': 0.85,
            'slot_goal_backoff': 0.55,
            'slot_goal_lateral': 0.75,
            'slot_tolerance': 0.55,
            'safe_target_radius': 0.18,
        }
        if reward_kwargs is not None:
            self.reward_cfg.update(reward_kwargs)

        self.controller_cfg = {
            'target_avoid_radius': 1.4,
            'target_avoid_gain': 1.4,
            'force_exponent': 2.0,
            'altitude_gain': 0.35,
        }
        if controller_kwargs is not None:
            self.controller_cfg.update(controller_kwargs)

        self._rng = np.random.default_rng(seed)
        self._goal_body_id = None
        self.hold_steps = 0
        self.prev_target_goal_dist = None
        self.last_metrics = {}
        self._metrics_step = None
        self._reward_step = None
        self._last_reward = 0.0

        num_drones = self.num_pursuers + 1

        if initial_xyzs is None:
            initial_xyzs = self._sample_initial_xyzs(self._rng)

        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )

        self.difficulty = 1.0

        self.discrete_actions = {
            0: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),

            # 1-8: horizontal plane, z = 0
            1:  np.array([ 1.0,        0.0,        0.0, self.pursuer_speed_fraction], dtype=np.float32),
            2:  np.array([ 1/np.sqrt(2),  1/np.sqrt(2),  0.0, self.pursuer_speed_fraction], dtype=np.float32),
            3:  np.array([ 0.0,        1.0,        0.0, self.pursuer_speed_fraction], dtype=np.float32),
            4:  np.array([-1/np.sqrt(2),  1/np.sqrt(2),  0.0, self.pursuer_speed_fraction], dtype=np.float32),
            5:  np.array([-1.0,        0.0,        0.0, self.pursuer_speed_fraction], dtype=np.float32),
            6:  np.array([-1/np.sqrt(2), -1/np.sqrt(2),  0.0, self.pursuer_speed_fraction], dtype=np.float32),
            7:  np.array([ 0.0,       -1.0,        0.0, self.pursuer_speed_fraction], dtype=np.float32),
            8:  np.array([ 1/np.sqrt(2), -1/np.sqrt(2),  0.0, self.pursuer_speed_fraction], dtype=np.float32),

            # 9-16: +45 degree inclination, upward
            9:  np.array([ 1/np.sqrt(2),  0.0,        1/np.sqrt(2), self.pursuer_speed_fraction], dtype=np.float32),
            10: np.array([ 0.5,        0.5,        1/np.sqrt(2), self.pursuer_speed_fraction], dtype=np.float32),
            11: np.array([ 0.0,        1/np.sqrt(2),  1/np.sqrt(2), self.pursuer_speed_fraction], dtype=np.float32),
            12: np.array([-0.5,        0.5,        1/np.sqrt(2), self.pursuer_speed_fraction], dtype=np.float32),
            13: np.array([-1/np.sqrt(2),  0.0,        1/np.sqrt(2), self.pursuer_speed_fraction], dtype=np.float32),
            14: np.array([-0.5,       -0.5,        1/np.sqrt(2), self.pursuer_speed_fraction], dtype=np.float32),
            15: np.array([ 0.0,       -1/np.sqrt(2),  1/np.sqrt(2), self.pursuer_speed_fraction], dtype=np.float32),
            16: np.array([ 0.5,       -0.5,        1/np.sqrt(2), self.pursuer_speed_fraction], dtype=np.float32),

            # 17-24: -45 degree inclination, downward
            17: np.array([ 1/np.sqrt(2),  0.0,       -1/np.sqrt(2), self.pursuer_speed_fraction], dtype=np.float32),
            18: np.array([ 0.5,        0.5,       -1/np.sqrt(2), self.pursuer_speed_fraction], dtype=np.float32),
            19: np.array([ 0.0,        1/np.sqrt(2), -1/np.sqrt(2), self.pursuer_speed_fraction], dtype=np.float32),
            20: np.array([-0.5,        0.5,       -1/np.sqrt(2), self.pursuer_speed_fraction], dtype=np.float32),
            21: np.array([-1/np.sqrt(2),  0.0,       -1/np.sqrt(2), self.pursuer_speed_fraction], dtype=np.float32),
            22: np.array([-0.5,       -0.5,       -1/np.sqrt(2), self.pursuer_speed_fraction], dtype=np.float32),
            23: np.array([ 0.0,       -1/np.sqrt(2), -1/np.sqrt(2), self.pursuer_speed_fraction], dtype=np.float32),
            24: np.array([ 0.5,       -0.5,       -1/np.sqrt(2), self.pursuer_speed_fraction], dtype=np.float32),

            # 25-26: pure vertical motion
            25: np.array([0.0, 0.0,  1.0, self.pursuer_speed_fraction], dtype=np.float32),
            26: np.array([0.0, 0.0, -1.0, self.pursuer_speed_fraction], dtype=np.float32),
        }

    def vec_to_action_mapper(self, vector):
        """
        Returns the action index and unit vector closest in direction to the input vector.
        """
        v = np.asarray(vector, dtype=np.float32)

        norm = np.linalg.norm(v)
        if norm < 1e-8:
            raise ValueError("Input vector has near-zero magnitude, cannot determine direction.")

        v_unit = v / norm

        best_action = None
        best_score = -np.inf

        for action_idx, action_vec in self.discrete_actions.items():
            score = np.dot(v_unit, action_vec[0:3])

            if score > best_score:
                best_score = score
                best_action = action_idx

        return best_action, self.discrete_actions[best_action]

    def _action_space(self,agent):
        """
        Learner only controls pursuers.
        The target drone's action is injected internally.
        """
        '''
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )
        '''
        return spaces.Discrete(27)

    def _observation_space(self,agent):

        idx = self.agents.index(agent)
        obs = self._computeObs()
        obs_dim = obs[idx].shape

        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_dim,
            dtype=np.float32,
        )

    def _computeObs(self):
        states = [self._getDroneStateVector(i) for i in range(self.NUM_DRONES)]
        positions = [s[0:3] for s in states]
        velocities = [s[10:13] for s in states]
        global_obs = []
        if self.global_observations:
            # Full-state mode: no estimate/filtering layer, just exact simulator state.
            global_obs = [
                np.asarray(positions, dtype=np.float32).reshape(-1),
                np.asarray(velocities, dtype=np.float32).reshape(-1),
                self.GOAL_POS.astype(np.float32),
            ]

        obs = []

        for i in range(self.num_pursuers):
            own_pos = positions[i]
            own_vel = velocities[i]
            goal_rel = self.GOAL_POS - own_pos

            target_rel = positions[self.target_idx] - own_pos

            other_rel = []
            for j in range(self.num_pursuers):
                if j == i:
                    continue
                other_rel.append(positions[j] - own_pos)

            obs_i = np.concatenate(
                [
                    own_vel,
                    own_pos,
                    goal_rel,
                    target_rel,
                    *other_rel,
                ]
            )

            obs.append(obs_i)

        return np.asarray(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.INIT_XYZS = self._sample_initial_xyzs(self._rng)
        self._sync_goal_marker()
        self.hold_steps = 0
        self.prev_target_goal_dist = None
        self.last_metrics = {}
        self._metrics_step = None
        self._reward_step = None
        self._last_reward = 0.0

        obs, info = super().reset(seed=seed, options=options)
        metrics = self.compute_team_metrics()
        self.prev_target_goal_dist = metrics['target_goal_dist']

        return self._convert_experience_to_dict(obs,info)

    def _sync_goal_marker(self):
        if self._goal_body_id is None:
            return
        p.resetBasePositionAndOrientation(
            self._goal_body_id,
            self.GOAL_POS.tolist(),
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.CLIENT,
        )

    def step(self, action_dict):

        action = []
        for agent in self.agents:
            action.append(self.discrete_actions[action_dict[agent]])


        action = np.asarray(action, dtype=np.float32)

        full_action = np.zeros((self.NUM_DRONES, 4), dtype=np.float32)
        full_action[: self.num_pursuers, :] = action
        full_action[self.target_idx, :] = self._scripted_target_action()

        obs, reward, terminated, truncated, info = super().step(full_action)

        return self._convert_experience_to_dict(obs, reward, terminated, truncated, info)

    def _convert_experience_to_dict(self, obs, *args):

        obs_dict = {
            agent: obs[i]
            for i, agent in enumerate(self.agents)
        }

        if len(args) == 1:
            info = args[0]

            info_dict = {
                agent: dict(info)
                for agent in self.agents
            } if isinstance(info, dict) else {
                agent: info
                for agent in self.agents
            }

            return obs_dict, info_dict

        if len(args) == 4:
            reward, terminated, truncated, info = args

            reward_dict = {
                agent: reward
                for agent in self.agents
            }

            terminated_dict = {
                agent: terminated
                for agent in self.agents
            }

            truncated_dict = {
                agent: truncated
                for agent in self.agents
            }

            info_dict = {
                agent: dict(info)
                for agent in self.agents
            } if isinstance(info, dict) else {
                agent: info
                for agent in self.agents
            }

            return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

        raise ValueError(
            "Expected either (obs, info) or "
            "(obs, reward, terminated, truncated, info)."
        )

    def _scripted_target_action(self):
        """
        Target flees from pursuers using inverse-square repulsion.

        Output format for ActionType.VEL:
            [vx_dir, vy_dir, vz_dir, speed_fraction]
        """
        states = [self._getDroneStateVector(i) for i in range(self.NUM_DRONES)]
        target_pos = states[self.target_idx][0:3]

        force = np.zeros(3, dtype=np.float32)

        for i in range(self.num_pursuers):
            pursuer_pos = states[i][0:3]
            # Match the 2D prey behavior: flee in XY, use Z only to stabilize altitude.
            diff = target_pos - pursuer_pos
            diff[2] = 0.0
            dist = np.linalg.norm(diff) + 1e-6
            local_gain = 1.0
            if dist < self.controller_cfg['target_avoid_radius']:
                local_gain *= self.controller_cfg['target_avoid_gain']
            force += local_gain * diff / (dist ** self.controller_cfg['force_exponent'])

        force[2] += self.controller_cfg['altitude_gain'] * (self.base_altitude - target_pos[2])
        force_norm = np.linalg.norm(force)

        if force_norm < self.target_force_threshold:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        direction = force / force_norm

        return np.array(
            [
                direction[0],
                direction[1],
                direction[2],
                self.speed_ratio*self.pursuer_speed_fraction,
            ],
            dtype=np.float32,
        )

    def _computeReward(self):
        current_step = getattr(self, 'step_counter', None)
        if self._reward_step == current_step:
            return self._last_reward

        metrics = self.compute_team_metrics()
        reward = self.compute_team_reward(metrics)
        if metrics['oob']:
            reward -= self.reward_cfg['oob_penalty']
        self._last_reward = float(reward)
        self._reward_step = current_step
        return self._last_reward

    def _computeTerminated(self):
        metrics = self.compute_team_metrics()
        return bool(metrics.get('success', False) or metrics.get('oob', False))

    def _computeTruncated(self):

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True

        return False

    def _computeInfo(self):
        target_pos = self._getDroneStateVector(self.target_idx)[0:3]
        metrics = dict(self.compute_team_metrics())

        info = {
            "target_pos": target_pos.copy(),
            "goal_pos": self.GOAL_POS.copy(),
            "target_goal_dist": float(metrics['target_goal_dist']),
            "success": bool(metrics.get('success', False)),
        }
        info.update(metrics)
        return info

    def _target_goal_distance(self):
        target_pos = self._getDroneStateVector(self.target_idx)[0:3]
        return np.linalg.norm(target_pos - self.GOAL_POS)

    def _out_of_bounds(self, positions=None):
        if positions is None:
            positions, _ = self._positions_and_velocities()

        for i, position in enumerate(positions):
            x, y, z = position[0:3]

            if i == self.target_idx:
                xy_bound = self.grid_size
            else:
                xy_bound = self.grid_size * 2.0

            if abs(x) >= xy_bound or abs(y) >= xy_bound:
                return True

            if z < 0.05 or z > 3.0:
                return True

        return False

    def _positions_and_velocities(self):
        states = [self._getDroneStateVector(i) for i in range(self.NUM_DRONES)]
        positions = np.asarray([state[0:3] for state in states], dtype=np.float32)
        velocities = np.asarray([state[10:13] for state in states], dtype=np.float32)
        return positions, velocities

    def compute_team_metrics(self):
        current_step = getattr(self, 'step_counter', None)
        if self._metrics_step == current_step and self.last_metrics:
            return self.last_metrics

        positions, _ = self._positions_and_velocities()
        pursuer_positions = positions[: self.num_pursuers]
        target_pos = positions[self.target_idx]
        goal_pos = self.GOAL_POS.astype(np.float32)

        goal_vec = goal_pos - target_pos
        target_goal_dist = float(np.linalg.norm(goal_vec))
        goal_vec_xy = goal_vec.copy()
        goal_vec_xy[2] = 0.0
        target_goal_planar_dist = float(np.linalg.norm(goal_vec_xy))
        pursuer_target_vecs = pursuer_positions - target_pos
        pursuer_target_vecs_xy = pursuer_target_vecs.copy()
        pursuer_target_vecs_xy[:, 2] = 0.0
        pursuer_target_dists = np.linalg.norm(pursuer_target_vecs_xy, axis=1)

        touch_penalty = float(
            np.mean(np.clip(self.reward_cfg['safe_target_radius'] - pursuer_target_dists, 0.0, None))
        )
        radius_error = np.abs(pursuer_target_dists - self.reward_cfg['ideal_radius'])
        # Smooth instead of clipped: the drones need a usable signal before
        # they already happen to sit on the perfect ring.
        ring_score = float(
            np.mean(
                np.exp(-radius_error / max(float(self.reward_cfg['radius_tolerance']), 1e-6))
            )
        )
        close_fraction = float(
            np.mean(pursuer_target_dists <= self.reward_cfg['surround_radius'])
        )

        coverage_score = 1.0
        if len(pursuer_target_vecs) > 1:
            # Keep the coverage check horizontal; vertical spacing gets noisy in PyBullet.
            angles = np.sort(np.arctan2(pursuer_target_vecs_xy[:, 1], pursuer_target_vecs_xy[:, 0]))
            wrapped_angles = np.concatenate((angles, [angles[0] + (2 * np.pi)]))
            gaps = np.diff(wrapped_angles)
            ideal_gap = (2 * np.pi) / len(pursuer_target_vecs)
            max_gap = float(np.max(gaps))
            gap_penalty = max(0.0, max_gap - ideal_gap)
            coverage_score = float(
                np.clip(
                    1.0 - gap_penalty / max((2 * np.pi) - ideal_gap, 1e-6),
                    0.0,
                    1.0,
                )
            )

        if target_goal_planar_dist > 1e-6:
            goal_dir = goal_vec_xy / target_goal_planar_dist
            alignments = []
            for pursuer_vec, pursuer_dist in zip(pursuer_target_vecs_xy, pursuer_target_dists):
                if pursuer_dist < 1e-6:
                    alignments.append(1.0)
                    continue
                alignments.append(np.dot(pursuer_vec / pursuer_dist, -goal_dir))
            push_alignment = float((np.mean(alignments) + 1.0) / 2.0)
        else:
            push_alignment = 1.0

        altitude_error = float(abs(target_pos[2] - self.base_altitude))
        altitude_score = float(np.clip(1.0 - altitude_error / 0.5, 0.0, 1.0))
        slot_score, slot_distance = self.compute_slot_score(
            pursuer_positions,
            target_pos,
            goal_pos,
            target_goal_dist,
        )

        hold = (
            target_goal_dist <= self.reward_cfg['hold_goal_radius']
            and close_fraction >= self.reward_cfg['hold_close_fraction']
            and coverage_score >= self.reward_cfg['hold_coverage_min']
            and ring_score >= self.reward_cfg['hold_ring_min']
            and slot_score >= self.reward_cfg['hold_slot_min']
        )
        if hold:
            self.hold_steps += 1
        else:
            self.hold_steps = 0

        control_score = max(
            slot_score,
            0.5 * (ring_score + close_fraction),
        )
        goal_center_score = float(
            np.clip(
                1.0 - (target_goal_dist / self.reward_cfg['goal_center_radius']),
                0.0,
                1.0,
            )
        )
        goal_lock_proximity = float(
            np.clip(
                1.0 - (target_goal_dist / self.reward_cfg['goal_lock_radius']),
                0.0,
                1.0,
            )
        )
        goal_lock_score = float(goal_lock_proximity * control_score)

        metrics = {
            'target_goal_dist': target_goal_dist,
            'target_goal_planar_dist': target_goal_planar_dist,
            'ring_score': ring_score,
            'close_fraction': close_fraction,
            'coverage_score': coverage_score,
            'push_alignment': push_alignment,
            'slot_score': slot_score,
            'slot_distance': slot_distance,
            'control_score': control_score,
            'goal_center_score': goal_center_score,
            'goal_lock_proximity': goal_lock_proximity,
            'goal_lock_score': goal_lock_score,
            'altitude_score': altitude_score,
            'avg_pursuer_target_dist': float(np.mean(pursuer_target_dists)),
            'touch_penalty': touch_penalty,
            'hold': hold,
            'hold_steps': self.hold_steps,
            'success': self.hold_steps >= int(self.reward_cfg['success_hold_steps']),
            'oob': self._out_of_bounds(positions),
        }
        self.last_metrics = metrics
        self._metrics_step = current_step
        return metrics

    def compute_slot_score(self, pursuer_positions, target_pos, goal_pos, target_goal_dist):
        pursuer_target_vecs = pursuer_positions - target_pos
        pursuer_target_vecs[:, 2] = 0.0
        avg_pursuer_dist = float(
            np.mean(np.linalg.norm(pursuer_target_vecs, axis=1))
        )
        slots = self.compute_slots(
            target_pos,
            goal_pos,
            target_goal_dist,
            avg_pursuer_target_dist=avg_pursuer_dist,
        )
        best_average_distance = np.inf
        for slot_perm in _permutations(range(len(slots))):
            distances = [
                np.linalg.norm(pursuer_positions[idx] - slots[slot_idx])
                for idx, slot_idx in enumerate(slot_perm)
            ]
            best_average_distance = min(best_average_distance, float(np.mean(distances)))
        tolerance = max(float(self.reward_cfg['slot_tolerance']), 1e-6)
        # Keep the slot signal alive even when the drones are nowhere near the
        # ideal formation; a hard zero made PPO blind early in training.
        score = float(np.exp(-best_average_distance / tolerance))
        return score, float(best_average_distance)

    def assign_slots(self, pursuer_positions, slots):
        remaining_slots = [slot.copy() for slot in slots]
        assigned_slots = []
        for pursuer_pos in pursuer_positions:
            slot_idx = int(
                np.argmin(
                    [np.linalg.norm(pursuer_pos - slot) for slot in remaining_slots]
                )
            )
            assigned_slots.append(remaining_slots.pop(slot_idx))
        return np.asarray(assigned_slots, dtype=np.float32)

    def compute_slots(
        self,
        target_pos,
        goal_pos,
        target_goal_dist,
        avg_pursuer_target_dist=None,
    ):
        goal_vec = goal_pos - target_pos
        forward_xy = goal_vec[:2]
        forward_norm = np.linalg.norm(forward_xy)
        goal_planar_dist = float(forward_norm)
        if forward_norm > 1e-6:
            forward = (
                np.array([forward_xy[0], forward_xy[1], 0.0], dtype=np.float32)
                / forward_norm
            )
        else:
            forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        lateral = np.array([-forward[1], forward[0], 0.0], dtype=np.float32)

        if (
            avg_pursuer_target_dist is not None
            and avg_pursuer_target_dist > self.reward_cfg['slot_far_distance']
        ):
            slots = np.array(
                [
                    target_pos - forward * self.reward_cfg['slot_funnel_backoff'],
                    target_pos
                    - forward * self.reward_cfg['slot_funnel_backoff']
                    + lateral * self.reward_cfg['slot_funnel_lateral'],
                    target_pos
                    - forward * self.reward_cfg['slot_funnel_backoff']
                    - lateral * self.reward_cfg['slot_funnel_lateral'],
                ],
                dtype=np.float32,
            )
        elif goal_planar_dist > self.reward_cfg['slot_gather_distance']:
            slots = np.array(
                [
                    target_pos - forward * self.reward_cfg['slot_gather_backoff'],
                    target_pos
                    - forward * self.reward_cfg['slot_gather_backoff']
                    + lateral * self.reward_cfg['slot_gather_lateral'],
                    target_pos
                    - forward * self.reward_cfg['slot_gather_backoff']
                    - lateral * self.reward_cfg['slot_gather_lateral'],
                ],
                dtype=np.float32,
            )
        elif goal_planar_dist > self.reward_cfg['slot_switch_distance']:
            # Same idea as the 2D version: one pusher behind the target, two flanks.
            slots = np.array(
                [
                    target_pos - forward * self.reward_cfg['slot_push_radius'],
                    target_pos
                    - forward * self.reward_cfg['slot_push_offset']
                    + lateral * self.reward_cfg['slot_flank_radius'],
                    target_pos
                    - forward * self.reward_cfg['slot_push_offset']
                    - lateral * self.reward_cfg['slot_flank_radius'],
                ],
                dtype=np.float32,
            )
        else:
            slots = np.array(
                [
                    goal_pos - forward * self.reward_cfg['slot_goal_backoff'],
                    goal_pos + lateral * self.reward_cfg['slot_goal_lateral'],
                    goal_pos - lateral * self.reward_cfg['slot_goal_lateral'],
                ],
                dtype=np.float32,
            )
        slots[:, 2] = self.base_altitude
        return slots

    def compute_team_reward(self, metrics):
        if self.prev_target_goal_dist is None:
            progress = 0.0
        else:
            progress = self.prev_target_goal_dist - metrics['target_goal_dist']
        self.prev_target_goal_dist = metrics['target_goal_dist']

        near_goal_weight = float(
            np.exp(-metrics['target_goal_dist'] / self.reward_cfg['goal_focus_temp'])
        )
        hold_fraction = min(
            metrics['hold_steps'] / max(float(self.reward_cfg['success_hold_steps']), 1.0),
            1.0,
        )
        containment_weight = 0.25 + (0.75 * near_goal_weight)
        # Do not pay much for a loose blob around the prey. The last run found
        # that local optimum, so this keeps formation reward for actual control.
        containment_score = float(
            np.clip(
                (0.5 * (metrics['ring_score'] + metrics['close_fraction']) - 0.35) / 0.65,
                0.0,
                1.0,
            )
        )
        coverage_score = float(
            np.clip(
                (metrics['coverage_score'] - 0.05) / 0.95,
                0.0,
                1.0,
            )
        )
        approach_score = metrics['push_alignment'] - 0.5
        control_score = metrics['control_score']

        reward = (
            self.reward_cfg['progress_scale'] * progress
            - self.reward_cfg['distance_scale'] * metrics['target_goal_dist']
            - self.reward_cfg['chase_scale'] * metrics['avg_pursuer_target_dist']
            - self.reward_cfg['touch_penalty_scale'] * metrics['touch_penalty']
            + self.reward_cfg['approach_scale'] * approach_score
            + self.reward_cfg['slot_scale'] * metrics['slot_score']
            + self.reward_cfg['goal_center_scale'] * metrics['goal_center_score']
            + self.reward_cfg['goal_lock_scale'] * metrics['goal_lock_score']
            - self.reward_cfg['uncontrolled_goal_scale'] * near_goal_weight * (1.0 - control_score)
            + self.reward_cfg['containment_scale'] * containment_weight * containment_score
            + self.reward_cfg['coverage_scale'] * containment_weight * coverage_score
            + self.reward_cfg['altitude_scale'] * (metrics['altitude_score'] - 1.0)
            - self.reward_cfg['step_cost']
        )
        if metrics['hold']:
            reward += self.reward_cfg['hold_scale'] * (1.0 + hold_fraction + metrics['goal_lock_score'])
        if metrics['success']:
            reward += self.reward_cfg['success_bonus']
        return float(reward)

    def _sample_initial_xyzs(self, rng):
        xyzs = np.zeros((self.num_pursuers + 1, 3), dtype=np.float32)

        z_min = max(0.3, self.base_altitude - self.spawn_altitude_jitter)
        z_max = self.base_altitude + self.spawn_altitude_jitter

        theta = rng.uniform(-np.pi, np.pi)
        radius = rng.uniform(0.0, self.target_spawn_radius)
        target_xy = radius * np.array([np.cos(theta), np.sin(theta)])
        target_z = rng.uniform(z_min, z_max)

        target_pos = np.array(
            [target_xy[0], target_xy[1], target_z],
            dtype=np.float32,
        )

        xyzs[self.target_idx] = target_pos

        radial_dir = target_xy / (np.linalg.norm(target_xy) + 1e-6)
        if np.linalg.norm(radial_dir) < 1e-6:
            angle = rng.uniform(-np.pi, np.pi)
            radial_dir = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

        if self.pursuer_same_spawn:
            pack_angle = rng.uniform(-np.pi, np.pi)
            pack_radius = rng.uniform(self.pursuer_spawn_min_radius, self.pursuer_spawn_max_radius)
            pack_xy = target_xy + pack_radius * np.array([np.cos(pack_angle), np.sin(pack_angle)])

        for i in range(self.num_pursuers):
            if self.pursuer_same_spawn:
                pursuer_xy = pack_xy + rng.uniform(
                    -self.pursuer_same_spawn_jitter,
                    self.pursuer_same_spawn_jitter,
                    size=2,
                )
            else:
                theta = rng.uniform(-np.pi, np.pi)

                rot_mat = np.array(
                    [
                        [np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)],
                    ],
                    dtype=np.float32,
                )

                radius = rng.uniform(self.pursuer_spawn_min_radius, self.pursuer_spawn_max_radius)
                offset_xy = rot_mat @ (radial_dir * radius)
                pursuer_xy = target_xy + offset_xy

            pursuer_xy = np.clip(
                pursuer_xy,
                -self.grid_size,
                self.grid_size,
            )

            pursuer_z = rng.uniform(z_min, z_max)

            xyzs[i] = np.array(
                [pursuer_xy[0], pursuer_xy[1], pursuer_z],
                dtype=np.float32,
            )

        if self._fixed_goal_pos:
            self.GOAL_POS = self.GOAL_POS.astype(np.float32)
        else:
            self.GOAL_POS = (
                np.array([0.0, 0.0, self.base_altitude], dtype=np.float32)
            )
            self.GOAL_POS[:2] += rng.uniform(
                -self.goal_noise,
                self.goal_noise,
                size=2,
            ).astype(np.float32)

        return xyzs

    def _addObstacles(self):
        """
        Adds a translucent green goal marker.
        This is visual only, it has no collision body.
        """
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.capture_radius/5,
            rgbaColor=[0.2, 0.85, 0.2, 0.25],
            physicsClientId=self.CLIENT,
        )

        self._goal_body_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=self.GOAL_POS.tolist(),
            physicsClientId=self.CLIENT,
        )

    def set_difficulty(self, difficulty):
        self.difficulty = float(difficulty)

    def render_rgb(
        self,
        width: int = 640,
        height: int = 480,
        camera_position: np.ndarray = None,
        target_position: np.ndarray = None,
    ):
        """
        Returns an RGB image from the PyBullet simulation without GUI rendering.

        Args:
            width, height: image resolution
            camera_position: (3,) camera position in world frame
            target_position: (3,) point camera looks at

        Returns:
            rgb image as (H, W, 3) uint8 numpy array
        """

        if camera_position is None:
            camera_position = np.array([3.0, 3.0, 2.0], dtype=np.float32)

        if target_position is None:
            target_position = np.array([0.0, 0.0, 0.5], dtype=np.float32)

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_position.tolist(),
            cameraTargetPosition=target_position.tolist(),
            cameraUpVector=[0, 0, 1],
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=float(width) / height,
            nearVal=0.1,
            farVal=100.0,
        )

        _, _, rgba, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER,  # works without GUI
        )

        rgba = np.reshape(rgba, (height, width, 4))
        rgb = rgba[:, :, :3].astype(np.uint8)

        return rgb

