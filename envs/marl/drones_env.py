"""
Predator-prey drone environment for gym-pybullet-drones.

Learner controls N pursuer drones.
The final drone is a scripted target drone.
Reward is shared and sparse:
    reward = 1 if target is within capture_radius of goal, else 0

Recommended action type:
    ActionType.VEL

For ActionType.VEL, actions are 4D:
    [vx_dir, vy_dir, vz_dir, speed_fraction]
"""

import numpy as np
import pybullet as p
from gymnasium import spaces

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
        episode_len_sec: int = 25,
        target_force_threshold: float = 1.5,
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
        print(learned_agent_list, agent_list)
        self.agents = learned_agent_list
        self.all_agents = agent_list
        self.num_pursuers = len(learned_agent_list)
        self.target_idx = self.num_pursuers
        self.grid_size = float(grid_size)
        self.capture_radius = float(capture_radius)
        self.base_altitude = float(base_altitude)
        self.speed_ratio = float(speed_ratio)
        self.EPISODE_LEN_SEC = int(episode_len_sec)
        self.target_force_threshold = float(target_force_threshold)

        self.GOAL_POS = (
            np.array([0.0, 0.0, self.base_altitude], dtype=np.float32)
            if goal_pos is None
            else np.asarray(goal_pos, dtype=np.float32)
        )

        self._rng = np.random.default_rng()
        self._goal_body_id = None

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

    def _action_space(self,agent):
        """
        Learner only controls pursuers.
        The target drone's action is injected internally.
        """
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

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

        obs = []

        for i in range(self.num_pursuers):
            own_pos = positions[i]
            own_vel = velocities[i]
            goal_rel = self.GOAL_POS - own_pos

            target_rel = positions[self.target_idx] - own_pos

            obs_i = np.concatenate(
                [
                    own_vel,
                    own_pos,
                    goal_rel,
                    target_rel,
                ]
            )

            obs.append(obs_i)

        return np.asarray(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.INIT_XYZS = self._sample_initial_xyzs(self._rng)

        obs, info = super().reset(seed=seed, options=options)

        return self._convert_experience_to_dict(obs,info)

    def step(self, action_dict):

        action = []
        for agent in self.agents:
            action.append(action_dict[agent])


        action = np.asarray(action, dtype=np.float32)

        if action.shape == (self.num_pursuers, 3):
            speed = np.ones((self.num_pursuers, 1), dtype=np.float32)
            action = np.hstack([action, speed])

        if action.shape != (self.num_pursuers, 4):
            raise ValueError(
                f"Expected action shape {(self.num_pursuers, 4)}, got {action.shape}."
            )

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
            diff = target_pos - pursuer_pos
            dist = np.linalg.norm(diff) + 1e-6
            force += diff / (dist**2)

        force[2] = 0.0
        force_norm = np.linalg.norm(force)

        if force_norm < self.target_force_threshold:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        direction = force / force_norm

        return np.array(
            [
                direction[0],
                direction[1],
                0.0,
                self.speed_ratio,
            ],
            dtype=np.float32,
        )

    def _computeReward(self):
        target_dist = self._target_goal_distance()

        if target_dist > self.grid_size:
            return 0.0

        if target_dist < self.capture_radius:
            return 1.0

        return 0.0

    def _computeTerminated(self):
        if self._out_of_bounds():
            return True
        else:
            return False

    def _computeTruncated(self):

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True

        return False

    def _computeInfo(self):
        target_pos = self._getDroneStateVector(self.target_idx)[0:3]
        target_goal_dist = self._target_goal_distance()

        return {
            "target_pos": target_pos.copy(),
            "goal_pos": self.GOAL_POS.copy(),
            "target_goal_dist": float(target_goal_dist),
            "success": bool(target_goal_dist < self.capture_radius),
        }

    def _target_goal_distance(self):
        target_pos = self._getDroneStateVector(self.target_idx)[0:3]
        return np.linalg.norm(target_pos - self.GOAL_POS)

    def _out_of_bounds(self):
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            x, y, z = state[0:3]

            if i == self.target_idx:
                xy_bound = self.grid_size
            else:
                xy_bound = self.grid_size * 2.0

            if abs(x) >= xy_bound or abs(y) >= xy_bound:
                return True

            if z < 0.05 or z > 3.0:
                return True

        return False

    def _sample_initial_xyzs(self, rng):
        xyzs = np.zeros((self.num_pursuers + 1, 3), dtype=np.float32)

        z_min = max(0.3, self.base_altitude - 0.5)
        z_max = self.base_altitude + 0.5

        target_xy = rng.uniform(-2.0, 2.0, size=2)
        target_z = rng.uniform(z_min, z_max)

        target_pos = np.array(
            [target_xy[0], target_xy[1], target_z],
            dtype=np.float32,
        )

        xyzs[self.target_idx] = target_pos

        radial_dir = target_xy / (np.linalg.norm(target_xy) + 1e-6)

        for i in range(self.num_pursuers):
            theta = rng.uniform(-np.pi, np.pi)

            rot_mat = np.array(
                [
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)],
                ],
                dtype=np.float32,
            )

            radius = rng.uniform(0.3, 0.7)
            offset_xy = rot_mat @ (radial_dir * radius)

            pursuer_xy = np.clip(
                target_xy + offset_xy,
                -self.grid_size,
                self.grid_size,
            )

            pursuer_z = rng.uniform(z_min, z_max)

            xyzs[i] = np.array(
                [pursuer_xy[0], pursuer_xy[1], pursuer_z],
                dtype=np.float32,
            )

        self.GOAL_POS = np.array([0.0, 0.0, self.base_altitude])  + np.array(rng.uniform(-0.1, 0.1, size=3))

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