import pygame
import gymnasium
import numpy as np
from copy import deepcopy
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
        seed: Optional[int] = None,
    ):

        self.env = mpeEnv
        self.env.reset(seed=seed)
        self.agents = agents

        self.obs = None
        self.seed = seed

        self.ts = 0

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def step(
        self, 
        action_dict: dict
    ):
        self.ts += 1

        action_dict['target'] = self.adversary_action('target')

        obs, rewards, terminations, truncations, infos = self.env.step(action_dict)
        self.obs = obs

        oob = self.out_of_bounds()
        if oob:
            for agent in terminations:
                terminations[agent] = True
            terminations["__all__"] = True

        terminations["__all__"] = all(terminations.values())
        truncations["__all__"] = all(truncations.values())
        
        obs.pop("target", None)
        rewards.pop('target', None)
        terminations.pop('target', None)
        truncations.pop('target', None)
        
        return self.obs, rewards, terminations, truncations, {a: {} for a in self.agents}

    def reset(self, *, seed=None, options=None):
        self.ts = 0

        if seed is None:
            obs, info = self.env.reset()
        else:
            obs, info = self.env.reset(seed=seed)
        #self.obs = obs[0]

        self.obs = obs

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

            force += diff / dist #dist**2

        norm = np.linalg.norm(force)

        if norm > 0:
            force = force / norm

        # choose dominant direction
        if np.linalg.norm(force) < 1.0:
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
    ):

        self.agents = agent_list
        self.n = len(agent_list)
        self.base_speed = base_speed
        self.speed_ratio = speed_ratio

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
            #landmark.state.p_pos = np.array([1.0, 0.0])
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

        return np.concatenate(obs)


env = make_env(ScenarioEnv)
parallel_env = parallel_wrapper_fn(env)