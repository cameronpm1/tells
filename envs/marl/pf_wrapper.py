import time
import torch
import gymnasium
import numpy as np
from copy import deepcopy
from gymnasium import spaces
from collections import OrderedDict
from typing import Any, Dict, Type, Optional, Union
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class PFWrapper(MultiAgentEnv):
    '''
    wrapper for multiagent envs
    env should handle dict format, wrapper is
    for rllib MultiAgentEnv inheritance
    '''

    def __init__(
            self,
            env,
            particle_filter,
            agent_control_function,
            target_control_function,
            dim: int = 2,
            eval: bool = False,
            belief_kwargs: Optional[dict] = None,
            noise: Optional[float] = None,
    ): 
        super().__init__()

        self.env = env
        self.dim = dim
        self.eval = eval
        self.obs_map = env.obs_map
        self.belief_kwargs = belief_kwargs
        self.agents = deepcopy(env.agents)
        self.possible_agents = deepcopy(env.agents)
        self.last_raw_reward = None
        self.noise = noise

        self.single_observation_spaces = {
            agent: self.env._observation_space(agent)
            for agent in self.agents
        }

        self.single_action_spaces = {
            agent: self.env._action_space(agent)
            for agent in self.agents
        }

        # disable deprecated paths
        self.observation_spaces = None
        self.action_spaces = None

        self.observation_space = None
        self.action_space = None

        self.particle_filters = {}
        self.switch_count = {}
        self.observing_agent = {}
        self.consecutive_agent_count = {}
        for agent in self.agents:
            dim = self.dim
            start_dict = {}
            for agent2 in self.agents:
                if agent != agent2:
                    start_dict[agent2] = np.zeros((dim,))
            target_start = np.zeros((dim,))
            self.particle_filters[agent] = particle_filter(
                obs_map = self.obs_map,
                agent_start_pos = start_dict,
                target_start_pos = target_start,
                agent_control_function = agent_control_function,
                target_control_function = target_control_function,
                **belief_kwargs
            )
            self.switch_count[agent] = 0
            self.observing_agent[agent] = 'target'
            self.consecutive_agent_count[agent] = 0

        
        self.switch_time = 1 #number of timesteps it takes to observe new agent
        self.min_confidence_agent = {}

        self._step = 0
        

    def get_observation_space(self, agent):
        return self.single_observation_spaces[agent]

    def get_action_space(self, agent):
        return self.single_action_spaces[agent]
    
    def step(self, action_dict):

        terminated_all = False
        truncated_all = False

        obs,rew,terminated,truncated,env_infos = self.env.step(action_dict)
        rew = dict(rew)

        if self.eval:
            infos = {'target': obs['target']}
        else:
            infos = {}
        for agent in self.agents:
            infos[agent] = dict(env_infos.get(agent, {}))

        '''
        
        for each cluster, propagate
        get observation
        give new pos observation
        update weights
        determine if should switch gaze
        start clock to new observation

        '''
        #propogate particle filter, and update observations appropriately 
        new_obs = deepcopy(obs)
        errors = []

        for agent in self.agents:
            pos = obs[agent][self.obs_map['self_pos']]
            self.particle_filters[agent].propagate_all(new_obs[agent])

            pf_obs = self.particle_filters[agent].get_observation()
            new_obs[agent][self.obs_map['target_pos']] = pf_obs['target']['pos'] - pos
            start = self.obs_map['team'].start

            for i,key in enumerate(pf_obs.keys()):
                if 'agent' in key:
                    new_obs[agent][start:start+self.dim] = pf_obs[key]['pos'] - pos
                    start += self.dim
                if self.switch_count[agent] == 0:
                    if i == 0:
                        self.min_confidence_agent[agent] = (key, pf_obs[key]['confidence'])
                    if pf_obs[key]['confidence'] < self.min_confidence_agent[agent][1]:
                        self.min_confidence_agent[agent] = (key, pf_obs[key]['confidence'])
            all_same = all(pf_obs[agent2]['confidence'] == self.min_confidence_agent[agent][1] for agent2 in pf_obs.keys())
            if all_same or self.consecutive_agent_count[agent] > 2:
                self.consecutive_agent_count[agent] = 0
                ran_agent = np.random.choice(list(pf_obs.keys()))
                self.min_confidence_agent[agent] = (ran_agent, pf_obs[ran_agent]['confidence'])

            if self.switch_count[agent] >= 0:
                if self.switch_count[agent] == self.switch_time or self.min_confidence_agent[agent][0] == self.observing_agent[agent]:
                    self.particle_filters[agent].update_observation(self.min_confidence_agent[agent][0],obs[self.min_confidence_agent[agent][0]][self.obs_map['self_pos']])
                    self.switch_count[agent] = 0
                    if self.min_confidence_agent[agent][0] == self.observing_agent[agent]:
                        self.consecutive_agent_count[agent] += 1
                    else:
                        self.consecutive_agent_count[agent] = 0
                    self.observing_agent[agent] = self.min_confidence_agent[agent][0]
                else:
                    self.switch_count[agent] += 1
            state_slice = slice(self.obs_map['target_pos'].start,self.obs_map['team'].stop)
            error = self.permutation_invariant_error(obs[agent][state_slice], new_obs[agent][state_slice])
            errors.append(error)
            rew[agent] = rew[agent] - error * self.env.reward_cfg['belief_dev_scale']

        avg_error = float(np.mean(errors)) if errors else 0.0

        #obs.pop("target", None)
        new_obs.pop("target", None)
        rew.pop('target', None)
        terminated.pop('target', None)
        truncated.pop('target', None)

        terminated["__all__"] = all(terminated.values())
        truncated["__all__"] = all(truncated.values())
        
        infos['__common__'] = {}
        infos['__common__']['raw_reward'] = sum(rew.values())
        infos['__common__']['belief_error'] = avg_error

        self.last_raw_reward = sum(rew.values())

        self.obs = obs
        self.new_obs = new_obs

        self._step += 1

        return new_obs,rew,terminated,truncated,infos
    
    def reset(self, **kwargs):
        obs,infos = {},{}
        obs,infos = self.env.reset(**kwargs)

        infos['__common__'] = {}
        infos['__common__']['raw_reward'] = 0.0
        self.last_raw_reward = 0.0

        for agent in self.agents:
            #start_dict = {}
            #for agent2 in self.agents:
            #    if agent != agent2:
            #        start_dict[agent2] = obs[agent2][self.obs_map['self_pos']] #+ np.random.normal(0, 0.1, 2)
            #target_start = obs['target'][self.obs_map['self_pos']]
            self.particle_filters[agent].reset(obs)
            self.switch_count[agent] = 0
            self.observing_agent[agent] = 'target'

        obs.pop('target', None)

        return obs,infos
    
    def close(self):
        self.env.unwrapped.close()

    '''
    def render_rgb(self):
        img = self.env.render_rgb()
        return img
    '''

    def set_difficulty(self, difficulty):
        self.env.set_difficulty(difficulty)

    def permutation_invariant_error(self, pred, target):
        '''
        pred:   (N, 6)
        target: (N, 6)

        Returns:
            scalar, sum over batch of minimum assignment distances
        '''

        # Reshape to (N, 2, 3)
        pred = pred[self.dim:].reshape(-1, 2, self.dim)
        target = target[self.dim:].reshape(-1, 2, self.dim)

        # Direct assignment distances
        direct = (
            np.linalg.norm(pred[:, 0] - target[:, 0], axis=1) +
            np.linalg.norm(pred[:, 1] - target[:, 1], axis=1)
        )

        # Swapped assignment distances
        swapped = (
            np.linalg.norm(pred[:, 0] - target[:, 1], axis=1) +
            np.linalg.norm(pred[:, 1] - target[:, 0], axis=1)
        )

        static = np.linalg.norm(pred[:self.dim] - target[:self.dim])

        # Take minimum per sample, then sum batch
        return np.minimum(direct, swapped).sum() + static

    def render_rgb_pp(self, show: bool = False, window_name: str = "PredatorPrey"):
        """
        Draw the predator-prey environment using self.obs, then overlay estimated
        other-agent locations from self.new_obs with semi-transparent markers.

        Expected:
            self.obs:
                dict mapping agent_name -> observation array

            self.new_obs:
                dict mapping observer_agent_name -> estimated observation array

        Assumption for self.new_obs:
            self.new_obs[observer_name] has the same structural layout as self.obs[observer_name],
            but the "other agent relative positions" section contains estimated relative positions.

        Observation layout:
            obs[0:2] = own velocity
            obs[2:4] = own absolute position
            obs[4:6] = goal position relative to own position
            obs[6:6 + 2 * num_other_agents] = other-agent relative positions
            obs[-num_predators:] = predator ID one-hot
        """
        import cv2
        import numpy as np

        if self.obs is None:
            raise ValueError("self.obs is None. Call reset() or step() before drawing.")

        if not isinstance(self.obs, dict):
            raise TypeError("Expected self.obs to be a dict mapping agent names to observations.")

        width = getattr(self, "width", 700)
        height = getattr(self, "height", 700)
        grid_size = getattr(self, "GRID_SIZE", 10)
        cam_scale = getattr(self, "cam_scale", grid_size + 0.15)
        dot_scale = getattr(self, "dot_scale", 0.5)

        frame = np.full((height, width, 3), 255, dtype=np.uint8)

        def world_to_pixel(pos):
            x, y = float(pos[0]), float(pos[1])

            # Match original pygame draw behavior
            y *= -1.0

            px = (x / cam_scale) * (width / 2.0)
            py = (y / cam_scale) * (height / 2.0)

            px += width / 2.0
            py += height / 2.0

            return int(px), int(py)

        def safe_pixel(pos):
            px, py = world_to_pixel(pos)
            px = int(np.clip(px, 0, width - 1))
            py = int(np.clip(py, 0, height - 1))
            return px, py

        def draw_circle(base_frame, world_pos, color_bgr, radius, thickness=-1):
            px, py = safe_pixel(world_pos)
            cv2.circle(base_frame, (px, py), radius, color_bgr, thickness)
            cv2.circle(base_frame, (px, py), radius, (0, 0, 0), 1)

        def agent_color_bgr(name):
            if "agent" in name:
                # Original predator RGB was roughly (70, 70, 170)
                # OpenCV uses BGR, so this appears blue.
                return (170, 70, 70)
            else:
                # Original target RGB was roughly (170, 70, 70)
                # OpenCV uses BGR, so this appears red.
                return (70, 70, 170)

        entity_size = 0.05
        radius = int(entity_size * 350 * dot_scale)
        radius = max(radius, 1)

        estimated_radius = max(radius - 2, 2)
        estimate_alpha = 0.35

        agent_names = list(self.obs.keys())
        predator_names = [name for name in agent_names if "agent" in name]
        num_id_entries = len(predator_names)

        true_positions = {}
        goal_pos = None

        # ----------------------------
        # Extract true positions
        # ----------------------------
        for name, obs in self.obs.items():
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)

            if obs.shape[0] < 6:
                raise ValueError(f"Observation for {name} is too short: shape={obs.shape}")

            own_pos = obs[2:4]
            goal_rel = obs[4:6]

            true_positions[name] = own_pos

            if goal_pos is None:
                goal_pos = own_pos + goal_rel

        # ----------------------------
        # Draw true goal
        # ----------------------------
        if goal_pos is not None:
            draw_circle(
                frame,
                goal_pos,
                color_bgr=(50, 170, 50),
                radius=radius,
            )

            gx, gy = safe_pixel(goal_pos)
            cv2.putText(
                frame,
                "goal",
                (gx + radius + 3, gy - radius - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # ----------------------------
        # Draw true agent positions
        # ----------------------------
        for name, pos in true_positions.items():
            draw_circle(
                frame,
                pos,
                color_bgr=agent_color_bgr(name),
                radius=radius,
            )

            px, py = safe_pixel(pos)
            cv2.putText(
                frame,
                str(name),
                (px + radius + 3, py - radius - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # ----------------------------
        # Overlay estimated positions from self.new_obs
        # ----------------------------
        if hasattr(self, "new_obs") and self.new_obs is not None:
            overlay = frame.copy()

            for observer_name, estimated_obs in self.new_obs.items():
                if observer_name not in true_positions:
                    continue

                estimated_obs = np.asarray(estimated_obs, dtype=np.float32).reshape(-1)

                if estimated_obs.shape[0] < 8:
                    continue

                observer_true_pos = true_positions[observer_name]

                other_names = [name for name in agent_names if name != observer_name]

                # Estimated relative positions start after:
                #   own velocity, own position, goal-relative position
                other_start = 6

                # Ignore final predator-ID one-hot entries
                if num_id_entries > 0:
                    other_end = estimated_obs.shape[0] - num_id_entries
                else:
                    other_end = estimated_obs.shape[0]

                estimated_relative_flat = estimated_obs[other_start:other_end]
                max_pairs = len(estimated_relative_flat) // 2
                num_to_draw = min(len(other_names), max_pairs)

                for i in range(num_to_draw):
                    estimated_rel_pos = estimated_relative_flat[2 * i: 2 * i + 2]

                    # Convert estimated relative position into absolute world position
                    estimated_abs_pos = observer_true_pos + estimated_rel_pos

                    estimated_entity_name = other_names[i]
                    estimated_color = agent_color_bgr(estimated_entity_name)

                    px, py = safe_pixel(estimated_abs_pos)

                    # Filled translucent estimate marker
                    cv2.circle(
                        overlay,
                        (px, py),
                        estimated_radius,
                        estimated_color,
                        -1,
                    )

                    # White center makes estimates visually distinct from true positions
                    cv2.circle(
                        overlay,
                        (px, py),
                        max(1, estimated_radius // 3),
                        (255, 255, 255),
                        -1,
                    )

                    # Thin outline
                    cv2.circle(
                        overlay,
                        (px, py),
                        estimated_radius,
                        (0, 0, 0),
                        1,
                    )

                    cv2.putText(
                        overlay,
                        f"est:{observer_name}->{estimated_entity_name}",
                        (px + estimated_radius + 3, py + estimated_radius + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.32,
                        (80, 80, 80),
                        1,
                        cv2.LINE_AA,
                    )

            frame = cv2.addWeighted(overlay, estimate_alpha, frame, 1.0 - estimate_alpha, 0)

        if show:
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

        return frame

    def render_rgb(self, show: bool = False, window_name: str = "DroneEnv"):
        """
        Custom 2D top-down RGB renderer.

        Shows:
            - goal boxes
            - true protector drones as blue quadrotor symbols
            - true target/adversary drone as a red quadrotor symbol
            - semi-transparent estimated drone locations from self.new_obs

        Expected particle-filter estimate format:
            self.new_obs:
                dict mapping observer_agent_name -> estimated observation array

        Default estimate parsing assumption:
            estimated_obs[6:] contains flattened relative positions of other drones.

        For drones, each relative position is assumed to be 3D:
            [dx, dy, dz]

        If your observation only stores 2D relative positions, this function will
        also handle [dx, dy] automatically.
        """
        import numpy as np

        width = int(getattr(self.env, "render_width", 700))
        height = int(getattr(self.env, "render_height", 700))

        img = np.full((height, width, 3), 245, dtype=np.uint8)

        team_state = self.env._get_team_state()
        drone_pos = team_state[:, 0:3]
        box_pos = self.env.box_state[: self.env.num_goal_boxes, 0:3]

        hx, hy, _ = self.env.goal_box_half_extents

        # ------------------------------------------------------------
        # Agent names
        # ------------------------------------------------------------
        if hasattr(self, "obs") and isinstance(self.obs, dict):
            agent_names = list(self.obs.keys())
        else:
            agent_names = [f"drone_{i}" for i in range(len(drone_pos))]

        name_to_idx = {name: i for i, name in enumerate(agent_names)}

        # ------------------------------------------------------------
        # Dynamic square view around boxes, true drones, and estimates
        # ------------------------------------------------------------
        all_x_parts = [
            drone_pos[:, 0],
            box_pos[:, 0] - hx,
            box_pos[:, 0] + hx,
        ]
        all_y_parts = [
            drone_pos[:, 1],
            box_pos[:, 1] - hy,
            box_pos[:, 1] + hy,
        ]

        # Include estimated positions in the view if available.
        estimated_abs_positions = []

        if hasattr(self, "new_obs") and self.new_obs is not None:
            for observer_name, estimated_obs in self.new_obs.items():
                if observer_name not in name_to_idx:
                    continue

                observer_idx = name_to_idx[observer_name]
                observer_true_pos = drone_pos[observer_idx, 0:3]

                estimated_obs = np.asarray(estimated_obs, dtype=np.float32).reshape(-1)
                other_names = [name for name in agent_names if name != observer_name]

                other_start = int(getattr(self, "estimate_other_start", 6))
                other_end = int(getattr(self, "estimate_other_end", estimated_obs.shape[0]))

                if other_start >= estimated_obs.shape[0]:
                    continue

                flat = estimated_obs[other_start:other_end]

                # Prefer 3D relative estimates, but fall back to 2D if needed.
                if len(flat) >= 3 * len(other_names):
                    dim = 3
                elif len(flat) >= 2 * len(other_names):
                    dim = 2
                else:
                    continue

                for i, other_name in enumerate(other_names):
                    rel = flat[dim * i: dim * i + dim]

                    if dim == 2:
                        est = observer_true_pos.copy()
                        est[0:2] = observer_true_pos[0:2] + rel[0:2]
                    else:
                        est = observer_true_pos + rel[0:3]

                    estimated_abs_positions.append(est)

        if estimated_abs_positions:
            est_arr = np.asarray(estimated_abs_positions, dtype=np.float32)
            all_x_parts.append(est_arr[:, 0])
            all_y_parts.append(est_arr[:, 1])

        all_x = np.concatenate(all_x_parts)
        all_y = np.concatenate(all_y_parts)

        pad = max(2.0, float(self.env.target_spawn_distance) * 0.25)
        xmin, xmax = float(np.min(all_x) - pad), float(np.max(all_x) + pad)
        ymin, ymax = float(np.min(all_y) - pad), float(np.max(all_y) + pad)

        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)
        span = max(xmax - xmin, ymax - ymin, 1e-6)

        xmin, xmax = xmid - span / 2.0, xmid + span / 2.0
        ymin, ymax = ymid - span / 2.0, ymid + span / 2.0

        def world_to_pixel(x, y):
            px = int((x - xmin) / (xmax - xmin) * (width - 1))
            py = int((ymax - y) / (ymax - ymin) * (height - 1))
            px = int(np.clip(px, 0, width - 1))
            py = int(np.clip(py, 0, height - 1))
            return px, py

        def draw_filled_circle(canvas, cx, cy, r, color):
            color = np.asarray(color, dtype=np.uint8)

            x0, x1 = max(0, cx - r), min(width - 1, cx + r)
            y0, y1 = max(0, cy - r), min(height - 1, cy + r)

            if x1 < x0 or y1 < y0:
                return

            yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
            canvas[y0:y1 + 1, x0:x1 + 1][mask] = color

        def draw_circle_outline(canvas, cx, cy, r, color, thickness=2):
            color = np.asarray(color, dtype=np.uint8)

            x0, x1 = max(0, cx - r), min(width - 1, cx + r)
            y0, y1 = max(0, cy - r), min(height - 1, cy + r)

            if x1 < x0 or y1 < y0:
                return

            yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
            dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
            mask = (dist2 <= r ** 2) & (dist2 >= (r - thickness) ** 2)
            canvas[y0:y1 + 1, x0:x1 + 1][mask] = color

        def draw_line(canvas, x0, y0, x1, y1, color, thickness=1):
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            dx = abs(x1 - x0)
            dy = -abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx + dy

            x, y = x0, y0

            while True:
                draw_filled_circle(canvas, x, y, thickness, color)

                if x == x1 and y == y1:
                    break

                e2 = 2 * err

                if e2 >= dy:
                    err += dy
                    x += sx

                if e2 <= dx:
                    err += dx
                    y += sy

        def draw_rectangle(canvas, x0, y0, x1, y1, fill, outline, thickness=2):
            fill = np.asarray(fill, dtype=np.uint8)
            outline = np.asarray(outline, dtype=np.uint8)

            left, right = sorted([x0, x1])
            top, bottom = sorted([y0, y1])

            left = max(0, min(width - 1, left))
            right = max(0, min(width - 1, right))
            top = max(0, min(height - 1, top))
            bottom = max(0, min(height - 1, bottom))

            canvas[top:bottom + 1, left:right + 1] = fill

            for t in range(thickness):
                if top + t <= bottom - t and left + t <= right - t:
                    canvas[top + t, left:right + 1] = outline
                    canvas[bottom - t, left:right + 1] = outline
                    canvas[top:bottom + 1, left + t] = outline
                    canvas[top:bottom + 1, right - t] = outline

        def draw_drone(canvas, cx, cy, radius, color):
            color = np.asarray(color, dtype=np.uint8)
            black = np.array([20, 20, 20], dtype=np.uint8)

            arm = max(8, int(radius * 2.5))
            rotor_r = max(3, int(radius * 0.65))
            line_thick = max(1, radius // 4)

            draw_line(canvas, cx - arm, cy, cx + arm, cy, black, line_thick + 1)
            draw_line(canvas, cx, cy - arm, cx, cy + arm, black, line_thick + 1)

            draw_line(canvas, cx - arm, cy, cx + arm, cy, color, line_thick)
            draw_line(canvas, cx, cy - arm, cx, cy + arm, color, line_thick)

            rotor_centers = [
                (cx - arm, cy),
                (cx + arm, cy),
                (cx, cy - arm),
                (cx, cy + arm),
            ]

            for rx, ry in rotor_centers:
                draw_circle_outline(canvas, rx, ry, rotor_r + 1, black, thickness=2)
                draw_circle_outline(canvas, rx, ry, rotor_r, color, thickness=2)

            draw_filled_circle(canvas, cx, cy, radius + 1, black)
            draw_filled_circle(canvas, cx, cy, radius, color)

        def drone_color(idx):
            if idx == self.env.target_idx:
                return np.array([220, 35, 35], dtype=np.uint8)
            return np.array([35, 90, 220], dtype=np.uint8)

        # ------------------------------------------------------------
        # Background grid
        # ------------------------------------------------------------
        grid_color = np.array([225, 225, 225], dtype=np.uint8)

        for x in np.linspace(xmin, xmax, 9):
            px, _ = world_to_pixel(x, ymin)
            img[:, px] = grid_color

        for y in np.linspace(ymin, ymax, 9):
            _, py = world_to_pixel(xmin, y)
            img[py, :] = grid_color

        # ------------------------------------------------------------
        # Draw goal boxes
        # ------------------------------------------------------------
        for goal_idx, point in enumerate(box_pos):
            x, y = float(point[0]), float(point[1])

            p0 = world_to_pixel(x - hx, y - hy)
            p1 = world_to_pixel(x + hx, y + hy)

            fill = np.array([170, 225, 170], dtype=np.uint8)
            outline = np.array([40, 120, 40], dtype=np.uint8)

            if goal_idx == getattr(self.env, "current_target_box_idx", -1):
                outline = np.array([190, 60, 30], dtype=np.uint8)

            draw_rectangle(
                img,
                p0[0], p0[1],
                p1[0], p1[1],
                fill=fill,
                outline=outline,
                thickness=3,
            )

        # ------------------------------------------------------------
        # Draw true drones
        # ------------------------------------------------------------
        base_altitude = max(float(getattr(self.env, "base_altitude", 0.5)), 1e-6)

        true_radii = []

        for i, pos in enumerate(drone_pos):
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            px, py = world_to_pixel(x, y)

            altitude_scale = 1.0 - 0.18 * ((z - base_altitude) / base_altitude)
            altitude_scale = float(np.clip(altitude_scale, 0.65, 1.25))

            radius = max(5, int(9 * altitude_scale))
            true_radii.append(radius)

            draw_drone(img, px, py, radius, drone_color(i))

        # ------------------------------------------------------------
        # Overlay estimated drone locations from self.new_obs
        # ------------------------------------------------------------
        if hasattr(self, "new_obs") and self.new_obs is not None:
            overlay = img.copy()

            estimate_alpha = float(getattr(self, "estimate_alpha", 0.35))
            estimate_other_start = int(getattr(self, "estimate_other_start", 6))

            for observer_name, estimated_obs in self.new_obs.items():
                if observer_name not in name_to_idx:
                    continue

                observer_idx = name_to_idx[observer_name]
                observer_true_pos = drone_pos[observer_idx, 0:3]

                estimated_obs = np.asarray(estimated_obs, dtype=np.float32).reshape(-1)
                other_names = [name for name in agent_names if name != observer_name]

                if estimate_other_start >= estimated_obs.shape[0]:
                    continue

                flat = estimated_obs[estimate_other_start:]

                if len(flat) >= 3 * len(other_names):
                    dim = 3
                elif len(flat) >= 2 * len(other_names):
                    dim = 2
                else:
                    continue

                for j, other_name in enumerate(other_names):
                    if other_name not in name_to_idx:
                        continue

                    other_idx = name_to_idx[other_name]
                    if 'target' in other_name:
                        rel = estimated_obs[self.obs_map['target_pos']]
                        color = np.array([220, 35, 35])
                    else:
                        rel = estimated_obs[self.obs_map['team']][3*j : 3*(j+1)]
                        color = np.array([35, 90, 20])

                    if dim == 2:
                        est_pos = observer_true_pos.copy()
                        est_pos[0:2] = observer_true_pos[0:2] + rel[0:2]
                    else:
                        est_pos = observer_true_pos + rel[0:3]

                    px, py = world_to_pixel(float(est_pos[0]), float(est_pos[1]))

                    if dim == 3:
                        z = float(est_pos[2])
                    else:
                        z = float(drone_pos[other_idx, 2])

                    altitude_scale = 1.0 - 0.18 * ((z - base_altitude) / base_altitude)
                    altitude_scale = float(np.clip(altitude_scale, 0.65, 1.25))

                    est_radius = max(4, int(7 * altitude_scale))

                    # Ghost estimate marker.
                    draw_drone(overlay, px, py, est_radius, color)

                    # White center makes estimates visually distinct from true drones.
                    draw_filled_circle(
                        overlay,
                        px,
                        py,
                        max(1, est_radius // 3),
                        np.array([255, 255, 255], dtype=np.uint8),
                    )

                    # Extra black outline.
                    draw_circle_outline(
                        overlay,
                        px,
                        py,
                        max(est_radius + 2, 5),
                        np.array([20, 20, 20], dtype=np.uint8),
                        thickness=1,
                    )

            img = (
                estimate_alpha * overlay.astype(np.float32)
                + (1.0 - estimate_alpha) * img.astype(np.float32)
            ).astype(np.uint8)

        if show:
            import cv2
            cv2.imshow(window_name, img[:, :, ::-1])
            cv2.waitKey(1)

        return img
