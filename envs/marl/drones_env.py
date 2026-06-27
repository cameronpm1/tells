"""
Multi-box drone defense environment for gym-pybullet-drones.

Learner controls N protector drones.
The final drone is a scripted adversarial drone.

Task:
    Five goal boxes are represented internally as propagating point states.
    Each box state is [x, y, z, vx, vy, vz].
    Current default box velocity is zero, so the boxes remain fixed unless
    nonzero box velocities are assigned later.

    Protector drones must hover over box points to mark them as protected.
    The adversary periodically retargets an unprotected box and moves using
    a single potential-field action function. That function takes the drone
    team state, box state, and attraction/repulsion parameters as inputs.

Recommended action type:
    ActionType.VEL

For ActionType.VEL, actions are 4D:
    [vx_dir, vy_dir, vz_dir, speed_fraction]
"""

import numpy as np
import pybullet as p
from copy import deepcopy
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class PredatorPreyAviary(BaseRLAviary):
    def __init__(
        self,
        agent_list: list[str],
        learned_agent_list: list[str],
        max_episode_length: int = 200,
        grid_size: float = 20.0,
        goal_pos: np.ndarray | None = None,
        goal_line_center: np.ndarray | None = None,
        num_goal_boxes: int = 5,
        num_protectors: int = 3,
        goal_box_spacing: float = 1.0,
        goal_box_half_extents: tuple[float, float, float] = (0.5, 0.5, 0.1),
        intrusion_radius: float = 0.3,
        adversary_replan_steps: int = 10,
        base_altitude: float = 0.5,
        base_speed: float = 8.0,
        speed_ratio: float = 0.4,
        protection_radius: float = 1.1,
        controller_kwargs: dict = None,
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
        self.full_agent_list = agent_list
        self.agents = learned_agent_list
        self.n_agents = len(self.agents)
        self.target_idx = len(self.agents)
        self.grid_size = grid_size
        self.base_altitude = base_altitude
        self.base_speed = base_speed
        self.speed_ratio = speed_ratio
        self.max_episode_length = max_episode_length

        self.num_goal_boxes = int(num_goal_boxes)
        self.goal_box_spacing = float(goal_box_spacing)
        self.goal_box_half_extents = tuple(float(v) for v in goal_box_half_extents)
        self.protection_radius = float(protection_radius)
        self.intrusion_radius = float(intrusion_radius)
        self.adversary_replan_steps = int(adversary_replan_steps)

        self.controller_cfg = {
            'adversary_repulsion_radius': 0.75,
            'adversary_repulsion_gain': 1.0,
            'adversary_attraction_gain': 1.0,
            'adversary_replan_steps': 10,
            'protection_radius': self.protection_radius,
        }
        if controller_kwargs is not None:
            self.controller_cfg.update(controller_kwargs)

        if goal_line_center is None:
            if goal_pos is None:
                center_xy = np.array([0.0, 0.0], dtype=np.float32)
            else:
                center_xy = np.asarray(goal_pos, dtype=np.float32)[0:2]
        else:
            center_xy = np.asarray(goal_line_center, dtype=np.float32)[0:2]

        self.goal_line_center_xy = center_xy.astype(np.float32)
        self.box_state = np.zeros((self.num_goal_boxes, 6), dtype=np.float32)
        self._initialize_box_state()

        self._rng = np.random.default_rng()
        self._goal_body_ids: list[int] = []
        self._policy_step_counter = 0
        self.current_target_box_idx = 0

        self.num_drones = self.n_agents + 1

        if initial_xyzs is None:
            initial_xyzs = self._sample_initial_xyzs(self._rng)

        super().__init__(
            drone_model=drone_model,
            num_drones=self.num_drones,
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

        self._step = 0
        self.difficulty = 1.0

        target_start = 6
        target_stop = target_start + (3 * self.num_goal_boxes) + 3
        team_stop = target_stop + 3 * (len(self.agents) - 1)
        target_obs_stop = target_stop + 3 * len(self.agents)
        self.obs_map = {
            'self': slice(0, 6),
            'target': slice(target_start, target_stop),
            'team': slice(target_stop, team_stop),
            'target_obs': slice(target_stop, target_obs_stop),
        }
        self.goal_rel = True

        self.action_library = {
            0:  np.array([0.0, 0.0, 0.0, 0.0]),

            1:  np.array([1.0, 0.0, 0.0, self.base_speed]),
            2:  np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0.0, self.base_speed]),
            3:  np.array([0.0, 1.0, 0.0, self.base_speed]),
            4:  np.array([-1 / np.sqrt(2), 1 / np.sqrt(2), 0.0, self.base_speed]),
            5:  np.array([-1.0, 0.0, 0.0, self.base_speed]),
            6:  np.array([-1 / np.sqrt(2), -1 / np.sqrt(2), 0.0, self.base_speed]),
            7:  np.array([0.0, -1.0, 0.0, self.base_speed]),
            8:  np.array([1 / np.sqrt(2), -1 / np.sqrt(2), 0.0, self.base_speed]),

            9:  np.array([0.0, 0.0, 1.0, self.base_speed]),
            10: np.array([0.0, 0.0, -1.0, self.base_speed]),
        }

    def _initialize_box_state(self):
        """
        Initialize box point states.

        Each row is:
            [x, y, z, vx, vy, vz]

        The default velocity is zero. Future experiments can assign nonzero
        velocities to self.box_state[:, 3:6] and the points will propagate.
        """
        center_offset = 0.5 * (self.num_goal_boxes * self.goal_box_half_extents[0] * 2 + self.goal_box_spacing * (self.num_goal_boxes - 1))
        
        for k in range(self.num_goal_boxes):
            x = self.goal_line_center_xy[0] + (k*(self.goal_box_half_extents[0]*2+self.goal_box_spacing) - center_offset)
            y = self.goal_line_center_xy[1]
            z = self.base_altitude
            self.box_state[k, 0:3] = np.array([x, y, z], dtype=np.float32)
            self.box_state[k, 3:6] = 0.0

    def _propagate_box_points(self, dt: float):
        """
        Propagate the box point states forward with constant velocity.

        The current default velocity is zero, so this is a no-op unless the
        box velocities are modified elsewhere.
        """
        self.box_state[:, 0:3] += self.box_state[:, 3:6] * float(dt)
        self._update_goal_visuals()

    def _update_goal_visuals(self):
        """Move the visual ground boxes to the x-y positions of the point states."""
        if not self._goal_body_ids:
            return

        z_center = self.goal_box_half_extents[2]

        for goal_idx, body_id in enumerate(self._goal_body_ids):
            point = self.box_state[goal_idx, 0:3]
            p.resetBasePositionAndOrientation(
                bodyUniqueId=body_id,
                posObj=[float(point[0]), float(point[1]), float(z_center)],
                ornObj=[0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.CLIENT,
            )

    def _get_team_state(self):
        """
        Return all drone states in a compact array.

        Rows 0 through num_agents - 1 are protector drones.
        The final row is the adversary.

        Each row is:
            [x, y, z, vx, vy, vz]
        """
        team_state = np.zeros((self.num_drones, 6), dtype=np.float32)

        for i in range(self.num_drones):
            state = self._getDroneStateVector(i)
            team_state[i, 0:3] = state[0:3]
            team_state[i, 3:6] = state[10:13]

        return team_state

    def _action_space(self,agent):
        """
        Learner only controls protector drones.
        The adversarial drone action is injected internally.
        """
        return spaces.Discrete(len(self.action_library))

    def _observation_space(self,agent):
        """
        Per-protector observation:
            own velocity, 3
            own position, 3
            relative position to each box point, 3 * num_goal_boxes
            relative position to the adversary, 3
            relative position of every other protector, 3 * (num_agents - 1)
        """
        obs_dim = (
            6
            + (3 * self.num_goal_boxes)
            + 3
            + 3 * (self.n_agents - 1)
        )

        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _computeObs(self):
        team_state = self._get_team_state()
        positions = team_state[:, 0:3]
        velocities = team_state[:, 3:6]

        obs = {}

        for i in range(self.num_drones):
            local_obs = {}
            own_pos = positions[i]
            own_vel = velocities[i]
            box_rel_positions = np.array([box[0:3] - own_pos for box in self.box_state]).flatten()
            local_obs['self'] = np.concatenate((own_vel, own_pos))

            rel_positions = []
            for j in range(self.num_drones):
                if j == self.target_idx:
                    local_obs['target'] = np.concatenate((box_rel_positions, positions[self.target_idx] - own_pos))
                elif j == i:
                    continue
                else:
                    rel_positions.append(positions[j] - own_pos)
            local_obs['team'] = np.array(rel_positions).flatten()

            obs[self.full_agent_list[i]] = np.concatenate((local_obs['self'], local_obs['target'], local_obs['team']))

        return obs
    
    def _computeInfo(self):
        protected = self._compute_protected_boxes()
        breached = bool(self._breached_box())
        oob = bool(self._out_of_bounds())
        protected_fraction = float(np.mean(protected))
        success = bool(protected_fraction >= 1.0 and not breached and not oob)

        info = {}
        for agent in self.agents:
            info[agent] = {
                'protected_boxes': int(np.sum(protected)),
                'protected_fraction': protected_fraction,
                'breached': breached,
                'oob': oob,
                'success': success,
                'target_box_idx': int(self.current_target_box_idx),
            }
        info['target'] = {
            'target_box_idx': int(self.current_target_box_idx),
            'breached': breached,
            'oob': oob,
        }

        return info

    def reset(self, seed=None, options=None):
        self._step = 0

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._policy_step_counter = 0
        self.current_target_box_idx = int(self._rng.integers(0, self.num_goal_boxes))
        self._initialize_box_state()
        self.INIT_XYZS = self._sample_initial_xyzs(self._rng)

        obs, infos = super().reset(seed=seed, options=options)
        self.obs = obs

        return deepcopy(obs), {}

    def step(self, action_dict):
        self._step += 1

        action = []
        for agent in self.agents:
            action.append(self.action_library[action_dict[agent]])

        #action = np.asarray(action, dtype=np.float32)
        #speed = np.ones((self.n_agents, 1), dtype=np.float32) * self.base_speed
        #action = np.hstack([action, speed])

        dt = 1.0 / float(getattr(self, "CTRL_FREQ", 30))
        self._propagate_box_points(dt)

        full_action = np.zeros((self.num_drones, 4), dtype=np.float32)
        full_action[: self.n_agents, :] = action
        full_action[self.target_idx, :] = self._scripted_target_action()

        self._policy_step_counter += 1

        obs, rewards, terminations, truncations, infos = super().step(full_action)
        self.obs = obs

        return deepcopy(obs), rewards, terminations, truncations, infos

    def _scripted_target_action(self):
        """
        Compute the adversary action from the compact team state and box state.
        """
        replan = self._policy_step_counter % self.controller_cfg['adversary_replan_steps'] == 0

        action, target_box_idx = self._compute_adversary_action_from_state(
            obs=self.obs['target'],
            obs_map=self.obs_map,
            current_target_box_idx=self.current_target_box_idx,
            replan=replan,
        )

        self.current_target_box_idx = int(target_box_idx)

        return action

    def _compute_adversary_action_from_state(
        self,
        obs,
        obs_map,
        current_target_box_idx: int,
        replan: bool = True,
    ):
        adversary_pos = obs[obs_map['self']][3:]
        team_pos = (obs[obs_map['target_obs']] + np.tile(adversary_pos, self.n_agents)).reshape(-1,3)
        box_pos = (obs[obs_map['target']][:-3] + np.tile(adversary_pos, int(len(obs[obs_map['target']])/3 - 1))).reshape(-1,3)

        protection_radius = self.controller_cfg['protection_radius']
        repulsion_radius = self.controller_cfg['adversary_repulsion_radius']
        repulsion_gain = self.controller_cfg['adversary_repulsion_gain']
        attraction_gain = self.controller_cfg['adversary_attraction_gain']

        selected_target_box_idx = int(current_target_box_idx)

        if replan or not (0 <= selected_target_box_idx < self.num_goal_boxes):
            dists = np.linalg.norm(
                box_pos[:, None, 0:3] - team_pos[None, :, 0:3],
                axis=2,
            )

            hovered = np.any((dists <= protection_radius), axis=1)
            unhovered_indices = np.where(~hovered)[0]

            if len(unhovered_indices) > 0:
                dists_to_adversary = np.linalg.norm(box_pos[unhovered_indices, 0:3] - adversary_pos[0:3],axis=1)
                selected_target_box_idx = int(unhovered_indices[np.argmin(dists_to_adversary)])

        target_pos = box_pos[selected_target_box_idx]
        force = np.zeros(3, dtype=np.float32)
        attraction = target_pos - adversary_pos
        force += float(attraction_gain) * attraction

        for protector in team_pos:
            diff = adversary_pos - protector
            diff[2] = 0.0

            dist = np.linalg.norm(diff) + 1e-6

            if dist <= repulsion_radius:
                force += float(repulsion_gain) * diff / (dist**2)

        speed = self.base_speed * self.speed_ratio

        if np.linalg.norm(force) < 0.5:
            action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            direction = force / np.linalg.norm(force)
            action = np.concatenate([direction, np.array([speed])])

        return action, selected_target_box_idx
    
    def _computeReward(self):
        """
        Shared team reward.

        Positive component:
            fraction of boxes currently protected.

        Negative component:
            penalty if the adversary breaches any box point.
        """
        protected = self._compute_protected_boxes()
        reward = float(np.mean(protected))

        if self._breached_box():
            reward -= 1.0

        reward_dict = {}
        for agent in self.agents:
            reward_dict[agent] = reward
        reward_dict['target'] = 0.0

        return reward_dict

    def _computeTerminated(self):

        if self._out_of_bounds():
            terminated = True
        elif self._breached_box():
            terminated = True
        else:
            terminated = False

        terminated_dict = {}
        for agent in self.agents:
            terminated_dict[agent] = terminated
        terminated_dict['target'] = False

        return terminated_dict

    def _computeTruncated(self):
        if self._step > self.max_episode_length:
            truncated = True
        else:
            truncated = False

        truncated_dict = {}
        for agent in self.agents:
            truncated_dict[agent] = truncated
        truncated_dict['target'] = False

        return truncated_dict

    def _compute_protected_boxes(self):
        team_state = self._get_team_state()
        protector_pos = team_state[: self.n_agents, 0:3]
        protected = np.zeros(self.num_goal_boxes, dtype=bool)

        for goal_idx, point in enumerate(self.box_state[:, 0:3]):
            for agent_idx in range(self.n_agents):
                pos = protector_pos[agent_idx]
                horizontal_dist = np.linalg.norm(pos - point)

                if horizontal_dist <= self.protection_radius:
                    protected[goal_idx] = True
                    break

        return protected

    def _breached_box(self):
        target_pos = self._getDroneStateVector(self.target_idx)[0:3]

        for goal_idx, point in enumerate(self.box_state[:, 0:3]):
            horizontal_dist = np.linalg.norm(target_pos[0:2] - point[0:2])

            if horizontal_dist <= self.intrusion_radius:
                return True

        return False

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

        for point in self.box_state[:, 0:3]:
            if abs(point[0]) >= self.grid_size * 2.0 or abs(point[1]) >= self.grid_size * 2.0:
                return True

        return False

    def _sample_initial_xyzs(self, rng):
        spawn_radius = 1.0
        xyzs = np.zeros((self.n_agents + 1, 3), dtype=np.float32)

        min_goal_x = float(np.min(self.box_state[:, 0]))
        max_goal_x = float(np.max(self.box_state[:, 0]))
        min_goal_y = float(np.min(self.box_state[:, 1]))
        max_goal_y = float(np.max(self.box_state[:, 1]))
        target_x = rng.uniform(min_goal_x, max_goal_x)
        target_y_sign = -1.0 if rng.random() < 0.5 else 1.0
        target_y = rng.uniform(min_goal_x, max_goal_x) + target_y_sign * 5

        xyzs[self.target_idx] = np.array(
            [target_x, target_y, self.base_altitude*2],
            dtype=np.float32,
        )

        if self.n_agents <= self.num_goal_boxes:
            assigned_goals = np.rint(
                np.linspace(0, self.num_goal_boxes - 1, self.n_agents)
            ).astype(int)
        else:
            assigned_goals = np.arange(self.n_agents) % self.num_goal_boxes

        box_idx = int(rng.integers(0, self.num_goal_boxes))
        box_center = self.box_state[box_idx, 0:3].astype(np.float32)

        for i in range(self.n_agents):
            angle = rng.uniform(-np.pi, np.pi)
            radius = rng.uniform(0.0, spawn_radius)

            offset_xy = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            xy = box_center[0:2] + offset_xy
            xyzs[i] = np.array([xy[0],xy[1], self.base_altitude])

        return xyzs

    def _addObstacles(self):
        """
        Adds translucent goal boxes on the ground.
        The actual target/protection state is the propagating box point state.
        These visual boxes do not create collision geometry.
        """
        self._goal_body_ids = []

        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=list(self.goal_box_half_extents),
            rgbaColor=[0.2, 0.85, 0.2, 0.35],
            physicsClientId=self.CLIENT,
        )

        z_center = self.goal_box_half_extents[2]

        for point in self.box_state[:, 0:3]:
            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=[float(point[0]), float(point[1]), float(z_center)],
                physicsClientId=self.CLIENT,
            )
            self._goal_body_ids.append(body_id)

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
            camera_position = np.array([4.5, 4.5, 2.9], dtype=np.float32)

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
