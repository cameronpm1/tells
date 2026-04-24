import pygame
import gymnasium
import numpy as np
from copy import deepcopy
from itertools import permutations
from typing import Optional
from gymnasium import spaces

from mpe2._mpe_utils.core import World
from mpe2._mpe_utils.core import Agent, Landmark
from mpe2._mpe_utils.scenario import BaseScenario
from mpe2._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

GRID_SIZE = 10

class PredatorPreyEnv(gymnasium.Env):

    '''
    gym environment containing mpe environment w/ additional functions
    '''

    def __init__(
        self,
        mpeEnv,
        agents,
        reward_kwargs: Optional[dict] = None,
        controller_kwargs: Optional[dict] = None,
        seed: Optional[int] = None,
    ):

        self.env = mpeEnv
        self.env.reset(seed=seed)
        self.agents = agents
        self.reward_cfg = {
            'distance_scale': 2.0,
            'chase_scale': 2.0,
            'progress_scale': 16.0,
            'approach_scale': 2.5,
            'containment_scale': 2.0,
            'coverage_scale': 2.0,
            'slot_scale': 18.0,
            'hold_scale': 20.0,
            'success_bonus': 750.0,
            'oob_penalty': 3000.0,
            'uncontrolled_goal_scale': 12.0,
            'touch_penalty_scale': 3.0,
            'step_cost': 0.05,
            'goal_focus_temp': 1.0,
            'surround_radius': 1.8,
            'ideal_radius': 1.5,
            'radius_tolerance': 0.55,
            'hold_goal_radius': 0.45,
            'hold_coverage_min': 0.45,
            'hold_close_fraction': 2.0 / 3.0,
            'hold_ring_min': 0.20,
            'success_hold_steps': 15,
            'slot_switch_distance': 3.0,
            'slot_push_radius': 2.1,
            'slot_push_offset': 1.05,
            'slot_flank_radius': 1.65,
            'slot_goal_backoff': 1.20,
            'slot_goal_lateral': 1.25,
            'slot_tolerance': 0.85,
        }
        if reward_kwargs is not None:
            self.reward_cfg.update(reward_kwargs)
        self.controller_cfg = {
            'action_threshold': 0.45,
            'force_exponent': 2.5,
            'boundary_margin': 0.75,
            'prey_sensitivity': 1.2,
            'prey_avoid_radius': 2.4,
            'prey_avoid_gain': 1.5,
        }
        if controller_kwargs is not None:
            self.controller_cfg.update(controller_kwargs)

        self.obs = None
        self.seed = seed

        self.ts = 0
        self.hold_steps = 0
        self.prev_target_goal_dist = None
        self.last_metrics = {}

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def step(
        self, 
        action_dict: dict
    ):
        self.ts += 1

        filtered_actions = {}
        for agent_id, action in action_dict.items():
            filtered_actions[agent_id] = self.boundary_safe_action(agent_id, action)
        filtered_actions['target'] = self.boundary_safe_action(
            'target',
            self.adversary_action('target'),
        )

        obs, rewards, terminations, truncations, infos = self.env.step(filtered_actions)
        self.obs = obs

        metrics = self.compute_team_metrics()
        team_reward = self.compute_team_reward(metrics)
        for agent in self.agents:
            rewards[agent] = team_reward

        if metrics['success']:
            for agent in terminations:
                terminations[agent] = True
            terminations["__all__"] = True

        oob = self.out_of_bounds()
        metrics['oob'] = oob
        if oob:
            for agent in self.agents:
                rewards[agent] -= self.reward_cfg['oob_penalty']
            for agent in terminations:
                terminations[agent] = True
            terminations["__all__"] = True

        terminations["__all__"] = all(terminations.values())
        truncations["__all__"] = all(truncations.values())
        
        obs.pop("target", None)
        rewards.pop('target', None)
        terminations.pop('target', None)
        truncations.pop('target', None)

        info = {
            agent: {
                'team_reward': team_reward,
                **metrics,
            }
            for agent in self.agents
        }

        return self.obs, rewards, terminations, truncations, info

    def reset(self, *, seed=None, options=None):
        self.ts = 0
        self.hold_steps = 0
        self.prev_target_goal_dist = None
        self.last_metrics = {}

        if seed is None:
            obs, info = self.env.reset()
        else:
            obs, info = self.env.reset(seed=seed)
        #self.obs = obs[0]

        self.obs = obs
        metrics = self.compute_team_metrics()
        self.prev_target_goal_dist = metrics['target_goal_dist']
        self.last_metrics = metrics

        obs.pop('target', None)
        return self.obs, {a: {} for a in self.agents}

    def get_obs(self):

        return self.obs

    def seed(
        self,
        seed:int
    ):

        self.seed = seed

    def adversary_action(
        self, 
        agent_id:str
    ):

        # find adversary entity
        world = self.env.unwrapped.world

        adversary = [a for a in world.agents if a.name == agent_id][0]

        force = np.zeros(2)

        for agent in world.agents:
            if agent.adversary:
                continue

            diff = adversary.state.p_pos - agent.state.p_pos
            dist = np.linalg.norm(diff) + 1e-6
            local_gain = self.controller_cfg['prey_sensitivity']
            if dist < self.controller_cfg['prey_avoid_radius']:
                local_gain *= self.controller_cfg['prey_avoid_gain']
            force += local_gain * (diff / dist ** self.controller_cfg['force_exponent'])

        norm = np.linalg.norm(force)

        if norm > 0:
            force = force / norm

        # choose dominant direction
        if norm < self.controller_cfg['action_threshold']:
            return 0  # no-op

        if abs(force[0]) > abs(force[1]):
            return 2 if force[0] > 0 else 1   # right / left
        else:
            return 4 if force[1] > 0 else 3   # up / down

    def render_rgb(self):
        """
        Returns an RGB frame of the current environment state
        """
        viewer = self.env.unwrapped.viewer

        if viewer is not None:
            viewer.set_bounds(-25.5, 25.5, -25.5, 25.5)  # fixed box
        frame = self.env.render()
        return frame

    def out_of_bounds(self, threshold=GRID_SIZE):
        world = self.env.unwrapped.world

        for entity in world.agents:
            x, y = entity.state.p_pos
            if abs(x) >= threshold or abs(y) >= threshold:
                return True

        return False

    def boundary_safe_action(self, agent_id: str, action: int):
        world = self.env.unwrapped.world
        agent = [a for a in world.agents if a.name == agent_id][0]
        x, y = agent.state.p_pos
        margin = self.controller_cfg['boundary_margin']

        if action == 1 and x <= (-GRID_SIZE + margin):
            return 0
        if action == 2 and x >= (GRID_SIZE - margin):
            return 0
        if action == 3 and y <= (-GRID_SIZE + margin):
            return 0
        if action == 4 and y >= (GRID_SIZE - margin):
            return 0

        return action

    def compute_team_metrics(self):
        world = self.env.unwrapped.world
        target = [a for a in world.agents if a.adversary][0]
        predators = [a for a in world.agents if not a.adversary]
        goal = world.landmarks[0]

        predator_positions = np.array([agent.state.p_pos for agent in predators])
        target_pos = target.state.p_pos.copy()
        goal_pos = goal.state.p_pos.copy()

        goal_vec = goal_pos - target_pos
        target_goal_dist = float(np.linalg.norm(goal_vec))

        predator_target_vecs = predator_positions - target_pos
        predator_target_dists = np.linalg.norm(predator_target_vecs, axis=1)
        touch_radius = 2.0 * predators[0].size
        touch_penalty = float(np.mean(np.clip(touch_radius - predator_target_dists, 0.0, None)))

        radius_error = np.abs(predator_target_dists - self.reward_cfg['ideal_radius'])
        ring_score = float(
            np.mean(
                np.clip(
                    1.0 - radius_error / self.reward_cfg['radius_tolerance'], #something like this?
                    0.0,
                    1.0,
                )
            )
        )
        close_fraction = float(
            np.mean(predator_target_dists <= self.reward_cfg['surround_radius'])
        )
        
        # ANGLE CALCULATION FOR COVERAGE

        coverage_score = 1.0
        if len(predator_target_vecs) > 1:
            # We only need a rough "are they wrapped around the prey?" signal here,
            # so the largest angular gap is a decent proxy for broken containment.
            angles = np.sort(np.arctan2(predator_target_vecs[:, 1], predator_target_vecs[:, 0]))
            wrapped_angles = np.concatenate((angles, [angles[0] + (2 * np.pi)]))
            gaps = np.diff(wrapped_angles)
            ideal_gap = (2 * np.pi) / len(predator_target_vecs)
            max_gap = float(np.max(gaps))
            gap_penalty = max(0.0, max_gap - ideal_gap)
            coverage_score = float(
                np.clip(
                    1.0 - gap_penalty / max((2 * np.pi) - ideal_gap, 1e-6),
                    0.0,
                    1.0,
                )
            )

        if target_goal_dist > 1e-6:
            goal_dir = goal_vec / target_goal_dist
            alignments = []
            for predator_vec, predator_dist in zip(predator_target_vecs, predator_target_dists):
                if predator_dist < 1e-6:
                    alignments.append(1.0)
                    continue
                alignments.append(np.dot(predator_vec / predator_dist, -goal_dir))
            push_alignment = float((np.mean(alignments) + 1.0) / 2.0)
        else:
            push_alignment = 1.0

        hold = (
            # "Hold" is stricter than "near the goal": the prey has to be near the
            # target and still look controlled by the formation.
            target_goal_dist <= self.reward_cfg['hold_goal_radius']
            and close_fraction >= self.reward_cfg['hold_close_fraction']
            and coverage_score >= self.reward_cfg['hold_coverage_min']
            and ring_score >= self.reward_cfg['hold_ring_min']
        )
        slot_score = self.compute_slot_score(
            predator_positions=predator_positions,
            target_pos=target_pos,
            goal_pos=goal_pos,
            target_goal_dist=target_goal_dist,
        )
        if hold:
            self.hold_steps += 1
        else:
            self.hold_steps = 0

        success = self.hold_steps >= int(self.reward_cfg['success_hold_steps'])

        metrics = {
            'target_goal_dist': target_goal_dist,
            'ring_score': ring_score,
            'close_fraction': close_fraction,
            'coverage_score': coverage_score,
            'push_alignment': push_alignment,
            'slot_score': slot_score,
            'avg_predator_target_dist': float(np.mean(predator_target_dists)),
            'touch_penalty': touch_penalty,
            'hold': hold,
            'hold_steps': self.hold_steps,
            'success': success,
        }
        self.last_metrics = metrics
        return metrics

    def compute_slot_score(
        self,
        predator_positions: np.ndarray,
        target_pos: np.ndarray,
        goal_pos: np.ndarray,
        target_goal_dist: float,
    ) -> float:
        goal_vec = goal_pos - target_pos
        if target_goal_dist > 1e-6:
            forward = goal_vec / target_goal_dist
        else:
            forward = np.array([1.0, 0.0])
        lateral = np.array([-forward[1], forward[0]])

        if target_goal_dist > self.reward_cfg['slot_switch_distance']:
            # Far away from goal we score a push-and-flank shape behind the prey.
            slots = np.array(
                [
                    target_pos - (forward * self.reward_cfg['slot_push_radius']),
                    target_pos
                    - (forward * self.reward_cfg['slot_push_offset'])
                    + (lateral * self.reward_cfg['slot_flank_radius']),
                    target_pos
                    - (forward * self.reward_cfg['slot_push_offset'])
                    - (lateral * self.reward_cfg['slot_flank_radius']),
                ]
            )
        else:
            # Near the goal we switch from "push it there" to "keep it there".
            slots = np.array(
                [
                    goal_pos - (forward * self.reward_cfg['slot_goal_backoff']),
                    goal_pos + (lateral * self.reward_cfg['slot_goal_lateral']),
                    goal_pos - (lateral * self.reward_cfg['slot_goal_lateral']),
                ]
            )

        best_average_distance = np.inf
        for slot_perm in permutations(range(len(slots))):
            distances = [
                np.linalg.norm(predator_positions[idx] - slots[slot_idx])
                for idx, slot_idx in enumerate(slot_perm)
            ]
            best_average_distance = min(best_average_distance, float(np.mean(distances)))

        return float(
            np.clip(
                1.0 - (best_average_distance / self.reward_cfg['slot_tolerance']),
                0.0,
                1.0,
            )
        )

    def compute_team_reward(self, metrics: dict):
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
        containment_score = (
            metrics['ring_score'] + metrics['close_fraction'] - 1.0
        )
        coverage_score = max(metrics['coverage_score'] - 0.5, 0.0)
        approach_score = metrics['push_alignment'] - 0.5
        control_score = max(
            metrics['slot_score'],
            0.5 * (metrics['ring_score'] + metrics['close_fraction']),
        )

        team_reward = (
            (self.reward_cfg['progress_scale'] * progress)
            - (self.reward_cfg['distance_scale'] * metrics['target_goal_dist'])
            - (self.reward_cfg['chase_scale'] * metrics['avg_predator_target_dist'])
            - (self.reward_cfg['touch_penalty_scale'] * metrics['touch_penalty'])
            + (self.reward_cfg['approach_scale'] * approach_score)
            + (self.reward_cfg['slot_scale'] * metrics['slot_score'])
            - (
                self.reward_cfg['uncontrolled_goal_scale']
                * near_goal_weight
                * (1.0 - control_score)
            )
            + (
                self.reward_cfg['containment_scale']
                * containment_weight
                * containment_score
            )
            + (
                self.reward_cfg['coverage_scale']
                * containment_weight
                * coverage_score
            )
            - self.reward_cfg['step_cost']
        )

        if metrics['hold']:
            team_reward += self.reward_cfg['hold_scale'] * (1.0 + hold_fraction)

        if metrics['success']:
            team_reward += self.reward_cfg['success_bonus']

        return float(team_reward)

class ScenarioEnv(SimpleEnv):

    '''
    container env for mpe scenario class
    '''

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "custom_mpe_env",
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self, 
        scenario_kwargs: dict, 
        max_cycles: int = 25,
        render_mode=None,
        continuous_actions=False, 
    ):

        self.render_mode = render_mode
        self.cam_scale = GRID_SIZE + 0.15
        self.dot_scale = 0.5

        scenario = PredatorPreyScenario(**scenario_kwargs)
        world = scenario.make_world()

        super().__init__(
            scenario=scenario,
            world=world,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            dynamic_rescaling=True,
        )

    def draw(self):
        # clear screen
        self.screen.fill((255, 255, 255))

        # ---- FIXED CAMERA RANGE ----
        cam_range = self.cam_scale  # constant, no zoom

        # update geometry and text positions
        text_line = 0

        for e, entity in enumerate(self.world.entities):
            # world position
            x, y = entity.state.p_pos

            # flip y-axis (match old pyglet behavior)
            y *= -1
            
            # ---- FIXED COORDINATE TRANSFORM ----
            # map world coordinates → pixel coordinates
            x = (x / cam_range) * (self.width / 2)
            y = (y / cam_range) * (self.height / 2)

            x += self.width / 2
            y += self.height / 2

            # ---- FIXED SIZE (no dynamic scaling) ----
            radius = entity.size * 350 * self.dot_scale

            # draw entity
            pygame.draw.circle(self.screen, entity.color * 200, (int(x), int(y)), int(radius))
            pygame.draw.circle(self.screen, (0, 0, 0), (int(x), int(y)), int(radius), 1)

            # optional bounds check (can remove if needed)
            assert (
                0 < x < self.width and 0 < y < self.height
            ), f"Coordinates {(x, y)} are out of bounds."

            # ---- TEXT RENDERING (unchanged) ----
            if isinstance(entity, Agent):
                if entity.silent:
                    continue

                if np.all(entity.state.c == 0):
                    word = "_"
                elif self.continuous_actions:
                    word = "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                else:
                    word = alphabet[np.argmax(entity.state.c)]

                message = entity.name + " sends " + word + "   "

                message_x_pos = self.width * 0.05
                message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)

                self.game_font.render_to(
                    self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
                )

                text_line += 1

class PredatorPreyScenario(BaseScenario):

    def __init__(
        self,
        agent_list: list[str],
        base_speed: float = 1.0,
        speed_ratio: float = 0.4,
        spawn_around_target: bool = False,
        predator_corner_spawn: bool = False,
        predator_same_spawn: bool = False,
        predator_same_spawn_jitter: float = 0.0,
        target_spawn_extent: float = 0.5,
        agent_spawn_min_radius: float = 1.5,
        agent_spawn_max_radius: float = 3.0,
        goal_spawn_min_radius: float = 1.0,
        goal_spawn_max_radius: float = 3.0,
    ):

        self.agents = agent_list
        self.n = len(agent_list)
        self.base_speed = base_speed
        self.speed_ratio = speed_ratio
        self.spawn_around_target = spawn_around_target
        self.predator_corner_spawn = predator_corner_spawn
        self.predator_same_spawn = predator_same_spawn
        self.predator_same_spawn_jitter = predator_same_spawn_jitter
        self.target_spawn_extent = target_spawn_extent
        self.agent_spawn_min_radius = agent_spawn_min_radius
        self.agent_spawn_max_radius = agent_spawn_max_radius
        self.goal_spawn_min_radius = goal_spawn_min_radius
        self.goal_spawn_max_radius = goal_spawn_max_radius

    def make_world(self):

        world = World()
        world.dim_c = 2

        world.agents = []

        # cooperative agents
        for agent_name in self.agents:
            if 'agent' in agent_name:
                agent = Agent()
                agent.name = agent_name
                agent.collide = True
                agent.silent = True
                agent.adversary = False
                agent.size = 0.05
                agent.accel = 3.0
                agent.max_speed = self.base_speed
                agent.color = np.array([0.35,0.35,0.85])
            else:
                agent = Agent()
                agent.name = agent_name
                agent.adversary = True
                agent.collide = True
                agent.silent = True
                agent.size = 0.05
                agent.accel = 3.0
                agent.max_speed = self.base_speed*self.speed_ratio
                agent.color = np.array([0.85,0.35,0.35])
            world.agents.append(agent)

        # landmark
        world.landmarks = []

        landmark = Landmark()
        landmark.name = "goal"
        landmark.collide = False
        landmark.movable = False
        landmark.color = np.array([0.25,0.85,0.25])

        world.landmarks.append(landmark)

        return world

    def reset_world(self, world, np_random):
        predators = [agent for agent in world.agents if not agent.adversary]
        target = [agent for agent in world.agents if agent.adversary][0]
        goal = world.landmarks[0]

        if self.predator_corner_spawn:
            corner_margin = 1.25
            corners = [
                np.array([-GRID_SIZE + corner_margin, -GRID_SIZE + corner_margin]),
                np.array([GRID_SIZE - corner_margin, -GRID_SIZE + corner_margin]),
                np.array([-GRID_SIZE + corner_margin, GRID_SIZE - corner_margin]),
            ]
            corner_order = np_random.permutation(len(corners))
            for predator, corner_idx in zip(predators, corner_order):
                predator.state.p_pos = corners[corner_idx] + np_random.uniform(-0.35, 0.35, world.dim_p)
                predator.state.p_vel = np.zeros(world.dim_p)
                predator.state.c = np.zeros(world.dim_c)

            target.state.p_pos = np_random.uniform(-1.25, 1.25, world.dim_p)
            target.state.p_vel = np.zeros(world.dim_p)
            target.state.c = np.zeros(world.dim_c)

            goal.state.p_pos = np.array([4.5, 4.5]) + np_random.uniform(-0.6, 0.6, world.dim_p)
            goal.state.p_vel = np.zeros(world.dim_p)
            return

        if self.spawn_around_target:
            target_position = np_random.uniform(
                -self.target_spawn_extent,
                +self.target_spawn_extent,
                world.dim_p,
            )
            if self.predator_same_spawn:
                pack_angle = np_random.uniform(0.0, 2.0 * np.pi)
                pack_distance = np_random.uniform(
                    self.agent_spawn_min_radius,
                    self.agent_spawn_max_radius,
                )
                pack_offset = pack_distance * np.array([np.cos(pack_angle), np.sin(pack_angle)])
                pack_position = np.clip(
                    target_position + pack_offset,
                    -GRID_SIZE + 1.0,
                    GRID_SIZE - 1.0,
                )
            else:
                predator_angles = np_random.uniform(
                    0.0,
                    2.0 * np.pi,
                    len(predators),
                )
                predator_distances = np_random.uniform(
                    self.agent_spawn_min_radius,
                    self.agent_spawn_max_radius,
                    predator_angles.shape[0],
                )
            predator_idx = 0

            for agent in world.agents:
                if agent.adversary:
                    agent.state.p_pos = target_position.copy()
                else:
                    if self.predator_same_spawn:
                        jitter = np_random.uniform(
                            -self.predator_same_spawn_jitter,
                            self.predator_same_spawn_jitter,
                            world.dim_p,
                        )
                        agent.state.p_pos = np.clip(
                            pack_position + jitter,
                            -GRID_SIZE + 1.0,
                            GRID_SIZE - 1.0,
                        )
                    else:
                        angle = predator_angles[predator_idx]
                        distance = predator_distances[predator_idx]
                        offset = distance * np.array([np.cos(angle), np.sin(angle)])
                        agent.state.p_pos = np.clip(
                            target_position + offset,
                            -GRID_SIZE + 1.0,
                            GRID_SIZE - 1.0,
                        )
                    predator_idx += 1

                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)

            for landmark in world.landmarks:
                goal_angle = np_random.uniform(0.0, 2.0 * np.pi)
                goal_distance = np_random.uniform(
                    self.goal_spawn_min_radius,
                    self.goal_spawn_max_radius,
                )
                offset = goal_distance * np.array([np.cos(goal_angle), np.sin(goal_angle)])
                landmark.state.p_pos = np.clip(
                    target_position + offset,
                    -GRID_SIZE + 1.0,
                    GRID_SIZE - 1.0,
                )
                landmark.state.p_vel = np.zeros(world.dim_p)
            return

        for agent in world.agents:
            if 'agent' in agent.name:
                agent.state.p_pos = np_random.uniform(-4, +4, world.dim_p)
            else:
                #adversary spawned in smaller area of map
                agent.state.p_pos = np_random.uniform(-0.5, +0.5, world.dim_p)

            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for landmark in world.landmarks:
            landmark.state.p_pos = np_random.uniform(-2, +2, (world.dim_p,))
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):

        if agent.adversary:
            return 0

        adversary = [a for a in world.agents if a.adversary][0]
        goal = world.landmarks[0]

        dist = np.linalg.norm(adversary.state.p_pos - goal.state.p_pos)

        return ((len(self.agents)-1)*GRID_SIZE)/np.clip(dist,0.1,GRID_SIZE)
        #return 1/np.clip(dist,0.1,GRID_SIZE)
        #return np.exp(5/(1+dist))

    def observation(self, agent, world):
        # ADD NOISE TO OBSERVATIONS OF OTHER AGENTS

        obs = []

        obs.append(agent.state.p_vel)
        obs.append(agent.state.p_pos)

        for landmark in world.landmarks:
            obs.append(landmark.state.p_pos - agent.state.p_pos)

        for other in world.agents:
            if other is agent:
                continue
                #obs.append(other.state.p_pos)
            obs.append(other.state.p_pos - agent.state.p_pos)

        predator_names = [name for name in self.agents if 'agent' in name]
        agent_id = np.zeros(len(predator_names), dtype=np.float32)
        if agent.name in predator_names:
            agent_id[predator_names.index(agent.name)] = 1.0
        obs.append(agent_id)

        return np.concatenate(obs)


env = make_env(ScenarioEnv)
parallel_env = parallel_wrapper_fn(env)
