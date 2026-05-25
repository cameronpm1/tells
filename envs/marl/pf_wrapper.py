import time
import torch
import gymnasium
import numpy as np
from copy import deepcopy
from gymnasium import spaces
from collections import OrderedDict
from typing import Any, Dict, Type, Optional, Union
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from envs.marl.particle_filter import PredatorPreyParticleFilter


class PFWrapper(MultiAgentEnv):
    '''
    wrapper for multiagent envs
    env should handle dict format, wrapper is
    for rllib MultiAgentEnv inheritance
    '''

    def __init__(
            self,
            env,
            eval: bool = False,
            belief_kwargs: Optional[dict] = None,
    ): 
        super().__init__()

        self.env = env
        self.eval = eval
        self.belief_kwargs = belief_kwargs
        self.agents = deepcopy(env.agents)
        self.possible_agents = deepcopy(env.agents)
        self.last_raw_reward = None

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
            dim = 2
            start_dict = {}
            for agent2 in self.agents:
                if agent != agent2:
                    start_dict[agent2] = np.zeros((dim,))
            prey_start = np.zeros((dim,))
            self.particle_filters[agent] = PredatorPreyParticleFilter(
                agent_start_pos = start_dict,
                prey_start_pos = prey_start,
            )
            self.switch_count[agent] = 0
            self.observing_agent[agent] = 'target'
            self.consecutive_agent_count[agent] = 0

        
        self.switch_time = 2 #number of timesteps it takes to observe new agent
        self.min_confidence_agent = {}

        self._step = 0
        

    def get_observation_space(self, agent):
        return self.single_observation_spaces[agent]

    def get_action_space(self, agent):
        return self.single_action_spaces[agent]
    
    def step(self, action_dict):

        terminated_all = False
        truncated_all = False

        obs,rew,terminated,truncated,_ = self.env.step(action_dict)
        rew = dict(rew)

        if self.eval:
            infos = {'target': obs['target']}
        else:
            infos = {}

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
        infos['__common__'] = {}
        for agent in self.agents:
            pos = obs[agent][2:4]
            goal = obs[agent][4:6] + pos
            self.particle_filters[agent].propagate_all(pos,goal)

            pf_obs = self.particle_filters[agent].get_observation()
            new_obs[agent][6:8] = pf_obs['target']['pos'] - pos
            start = 8

            for i,key in enumerate(pf_obs.keys()):
                # print(pf_obs)
                if 'agent' in key:
                    new_obs[agent][start:start+2] = pf_obs[key]['pos'] - pos
                    start += 2
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
                    self.particle_filters[agent].update_observation(self.min_confidence_agent[agent][0],obs[self.min_confidence_agent[agent][0]][2:4])
                    self.switch_count[agent] = 0
                    if self.min_confidence_agent[agent][0] == self.observing_agent[agent]:
                        self.consecutive_agent_count[agent] += 1
                    else:
                        self.consecutive_agent_count[agent] = 0
                    self.observing_agent[agent] = self.min_confidence_agent[agent][0]
                else:
                    self.switch_count[agent] += 1
            error = self.permutation_invariant_error(obs[agent][6:12], new_obs[agent][6:12])
            errors.append(error)
            #print(error)
        avg_error = np.average(errors)

        #obs.pop("target", None)
        #new_obs.pop("target", None)
        rew.pop('target', None)
        terminated.pop('target', None)
        truncated.pop('target', None)

        terminated["__all__"] = all(terminated.values())
        truncated["__all__"] = all(truncated.values())
        #print(obs.keys(),rew.keys(),terminated.keys(),truncated.keys(),_.keys())
        
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
            start_dict = {}
            for agent2 in self.agents:
                if agent != agent2:
                    start_dict[agent2] = obs[agent2][2:4] #+ np.random.normal(0, 0.1, 2)
            prey_start = obs['target'][2:4]
            self.particle_filters[agent].reset(
                agent_start_pos = start_dict,
                prey_start_pos = prey_start,
            )
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
        pred = pred[2:].reshape(-1, 2, 2)
        target = target[2:].reshape(-1, 2, 2)

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

        static = np.linalg.norm(pred[:2] - target[:2])

        # Take minimum per sample, then sum batch
        return np.minimum(direct, swapped).sum() + static

    def render_rgb(self, show: bool = False, window_name: str = "PredatorPrey"):
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








#good, not there is a new object called self.new_obs. for each key in self.new_obs the observation is an estimation of the agent locations other than the agent that correlates to self.new_obs key. create me an updated function that plots these estimated locations on the same map with a slightly opaque color.