
import gymnasium
import matplotlib
import numpy as np
from copy import deepcopy
from typing import Optional
from gymnasium import spaces
import matplotlib.pyplot as plt

matplotlib.use("Agg")  

from controllers.fire_control import probabilistic_fire_controller

GRID_SIZE = 60

class DroneFireEnv(gymnasium.Env):

    def __init__(
        self,
        drone_names: list[str],
        fire_controller,
        drone_start_area: list[int] = [0, 0],
        controller_kwargs: Optional[dict] = None,
        reward_kwargs: Optional[dict] = None,
        observation_range: int = 20,
        local_observations: bool = False,
        warm_up_steps: int = 10,
        local_radius: int = 5,
        max_episode_steps: int = 300,
        drone_start_noise: int = 3,
        ignition_points=None,
        n_random_fires: int = 1,
        extinguish_radius: int = 5,
    ):
        super().__init__()

        self.agents = list(drone_names)
        self.n_agents = len(self.agents)
        self.full_agent_list = self.agents + ['target']

        self.warm_up_steps = warm_up_steps
        self.observation_range = observation_range
        self.local_observations = local_observations
        self.local_radius = int(local_radius)
        self.max_episode_length = int(max_episode_steps)

        self.reward_kwargs = reward_kwargs if reward_kwargs is not None else {}
        self.controller_kwargs = controller_kwargs if controller_kwargs is not None else {}
        self.ignition_points = None if ignition_points is None else np.asarray(ignition_points, dtype=np.int32)
        self.drone_start_area = np.asarray(drone_start_area, dtype=np.int32)

        self.env = DroneFireSim(
            drone_names=drone_names,
            drone_start_area=drone_start_area,
            fire_controller=fire_controller,
            drone_start_noise=drone_start_noise,
            ignition_points=ignition_points,
            n_random_fires=n_random_fires,
            extinguish_radius=extinguish_radius,
        )

        self._step = 0
        self.obs = None

    def _observation_space(self, agent):

        return spaces.Box(
            low=0,
            high=3,
            shape=((self.observation_range * 2 + 1)**2,),
            dtype=np.int8,
        )

    def _action_space(self, agent):
        return spaces.Discrete(len(self.env.ACTION_LIBRARY))

    def _sample_center_quarter_ignition_points(self, n_points=3):
        lo, hi = GRID_SIZE // 4, 3 * GRID_SIZE // 4
        pts = []
        while len(pts) < n_points:
            pt = np.array([np.random.randint(lo, hi), np.random.randint(lo, hi)], dtype=np.int32)
            if not any(np.array_equal(pt, p) for p in pts):
                pts.append(pt)
        return np.asarray(pts, dtype=np.int32)

    def _sample_drone_start_area(self, ignition_points):
        center = ignition_points.mean(axis=0)
        for _ in range(100):
            angle = np.random.uniform(0.0, 2.0 * np.pi)
            radius = np.random.uniform(10.0, 15.0)
            candidate = np.round(center + radius * np.array([np.cos(angle), np.sin(angle)])).astype(np.int32)
            if 0 <= candidate[0] < GRID_SIZE and 0 <= candidate[1] < GRID_SIZE:
                return candidate
        return np.clip(np.round(center + np.array([15.0, 15.0])).astype(np.int32), 0, GRID_SIZE - 1)

    def reset(self, *, seed=None, options=None):
        self._step = 0

        if seed is not None:
            np.random.seed(seed)

        self.ignition_points = self._sample_center_quarter_ignition_points()
        self.drone_start_area = self._sample_drone_start_area(self.ignition_points)

        obs = self.env.reset(
            drone_start_area=self.drone_start_area,
            ignition_points=self.ignition_points,
        )
        self.obs = self._compute_obs(obs)

        for _ in range(self.warm_up_steps):
            actions = {agent: 0 for agent in self.agents}
            obs, rew, done, info = self.env.step(actions)

        info = {agent: {} for agent in self.agents}

        return deepcopy(self.obs), info

    def step(self, action_dict: dict):
        self._step += 1

        fire_before_actions = self.env.fire_state.copy()

        filtered_actions = {}
        for agent in self.agents:
            action = action_dict.get(agent, 0)
            filtered_actions[agent] = self.boundary_safe_action(agent, action)

        obs, rew, done, info = self.env.step(filtered_actions)

        truncations = self._compute_truncation()
        terminations = done
        #reward = self._compute_reward()

        info = {}
        self.obs = self._compute_obs(obs)
        reward = {agent : rew for agent in self.agents}
        reward['target'] = 0.0
        truncations = {agent : truncations for agent in self.full_agent_list}
        terminations = {agent : terminations or done for agent in self.full_agent_list}
        info = {agent : info for agent in self.agents}

        return deepcopy(self.obs), reward, terminations, truncations, info

    def _compute_obs(self, obs):
        obs_dict = {}
        window_size = self.observation_range * 2 + 1

        for agent in self.agents:
            agent_pos = obs['drones'][agent]
            partial_obs = self._local_fire_window(obs['fire'], agent_pos)

            for agent2 in self.agents:
                if agent2 == agent:
                    continue

                other_pos = obs['drones'][agent2]
                rel_r = other_pos[0] - agent_pos[0]
                rel_c = other_pos[1] - agent_pos[1]

                if abs(rel_r) <= self.observation_range and abs(rel_c) <= self.observation_range:
                    cross_r = rel_r + self.observation_range
                    cross_c = rel_c + self.observation_range
                    for rr, cc in [(cross_r, cross_c), (cross_r - 1, cross_c), (cross_r + 1, cross_c), (cross_r, cross_c - 1), (cross_r, cross_c + 1)]:
                        if 0 <= rr < window_size and 0 <= cc < window_size:
                            partial_obs[rr, cc] = 3.0

            agent_obs = partial_obs.flatten().astype(np.int8)
            obs_dict[agent] = agent_obs

        obs_dict['target'] = {'target': self.env.fire_state}

        return obs_dict

    def _compute_reward(
        self,
    ):
        return 0.0

    def _compute_truncation(self):

        if self._step >= self.max_episode_length:
            return True
        else:
            return False

    def _local_fire_window(self, fire_state, agent_pos):
        r, c = agent_pos
        radius = self.observation_range

        padded_fire = np.pad(
            fire_state,
            pad_width=radius,
            mode="constant",
            constant_values=self.env.GREEN,
        )

        rp = r + radius
        cp = c + radius

        window = padded_fire[
            rp - radius: rp + radius + 1,
            cp - radius: cp + radius + 1,
        ]

        return window.astype(np.float32)

    def boundary_safe_action(self, agent_id: str, action: int):
        action = int(action)

        if action not in self.env.ACTION_LIBRARY:
            return 0

        if action == 9:
            return 9

        pos = self.env.drone_states[agent_id]
        next_pos = pos + self.env.ACTION_LIBRARY[action]

        r, c = next_pos

        if r < 0 or r >= GRID_SIZE:
            return 0

        if c < 0 or c >= GRID_SIZE:
            return 0

        return action

    def get_obs(self):
        return deepcopy(self.obs)

    def seed(self, seed: int):
        self.seed_value = seed
        np.random.seed(seed)

        
    def render_rgb(self):
        frame = self.env.image().copy()
        blue = np.array([0, 0, 255], dtype=np.uint8)

        for drone in self.agents:
            r, c = self.env.drone_states[drone]
            for rr, cc in [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                if 0 <= rr < frame.shape[0] and 0 <= cc < frame.shape[1]:
                    frame[rr, cc] = blue

        return frame

class DroneFireSim:
    GREEN, RED, BLACK = 0, 1, 2

    COLORS = np.array([
        [0, 255, 0],    # green
        [255, 0, 0],    # red
        [0, 0, 0],      # black
    ], dtype=np.uint8)

    ACTION_LIBRARY = {
        0: np.array([0, 0]),     # stay
        1: np.array([-1, 0]),    # up
        2: np.array([1, 0]),     # down
        3: np.array([0, -1]),    # left
        4: np.array([0, 1]),     # right
        5: np.array([-1, -1]),   # up-left
        6: np.array([-1, 1]),    # up-right
        7: np.array([1, -1]),    # down-left
        8: np.array([1, 1]),     # down-right
        9: np.array([0, 0]),     #extinguish fire
    }

    def __init__(
        self,
        drone_names: list[str],
        drone_start_area: list[float],
        fire_controller,
        drone_start_noise: int = 3,
        ignition_points=None,
        n_random_fires: int = 1,
        extinguish_radius: int = 5,
    ):
        self.drone_names = drone_names
        self.n_drones = len(drone_names)
        self.drone_start_area = np.array(drone_start_area, dtype=np.int32)
        self.drone_start_noise = drone_start_noise
        self.fire_controller = fire_controller
        self.extinguish_radius = extinguish_radius

        self.ignition_points = ignition_points
        self.n_random_fires = n_random_fires

        self.drone_states = {}
        self.fire_state = None
        self.t = 0

        self.reset()

    def initialize_fire(self):
        fire = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

        if self.ignition_points is None:
            points = np.random.randint(
                0,
                GRID_SIZE,
                size=(self.n_random_fires, 2),
            )
        else:
            points = np.asarray(self.ignition_points, dtype=np.int32)

            if points.ndim == 1:
                points = points[None, :]

        points = np.clip(points, 0, GRID_SIZE - 1)
        fire[points[:, 0], points[:, 1]] = self.RED

        return fire

    def reset(
        self,
        drone_start_area: Optional[list[float]] = None,
        ignition_points: Optional[list[list[float]]] = None,
    ):
        if ignition_points is not None:
            self.ignition_points = np.asarray(ignition_points, dtype=np.int32)
            if self.ignition_points.ndim == 1:
                self.ignition_points = self.ignition_points[None, :]

        if drone_start_area is not None:
            self.drone_start_area = np.asarray(drone_start_area, dtype=np.int32)

        for drone in self.drone_names:
            noise = np.random.randint(-self.drone_start_noise, self.drone_start_noise + 1, size=2)
            self.drone_states[drone] = np.clip(self.drone_start_area + noise, 0, GRID_SIZE - 1)

        self.fire_state = self.initialize_fire()
        self.t = 0

        return self.state()

    def state(self):
        return {
            "t": self.t,
            "fire": self.fire_state.copy(),
            "drones": {k: v.copy() for k, v in self.drone_states.items()},
        }

    def extinguish_fire(
        self,
        drone,
        max_extinguish_prob=1.0,
        decay_rate=0.05,
    ):
        """
        Extinguish nearby fire cells probabilistically.

        Probability decreases with distance from the drone:
            p = max_extinguish_prob * exp(-decay_rate * distance)

        Cells changed:
            RED -> BLACK with probability p
            GREEN stays GREEN
            BLACK stays BLACK
        """

        pos = self.drone_states[drone]
        r, c = pos

        rows, cols = self.fire_state.shape

        r_min = max(0, r - self.extinguish_radius)
        r_max = min(rows, r + self.extinguish_radius + 1)
        c_min = max(0, c - self.extinguish_radius)
        c_max = min(cols, c + self.extinguish_radius + 1)

        extinguished_cells = 0

        for rr in range(r_min, r_max):
            for cc in range(c_min, c_max):
                distance = np.sqrt((rr - r) ** 2 + (cc - c) ** 2)

                if distance > self.extinguish_radius:
                    continue

                if self.fire_state[rr, cc] == self.RED:
                    extinguish_prob = max_extinguish_prob * np.exp(
                        -decay_rate * distance
                    )

                    if np.random.random() < extinguish_prob:
                        self.fire_state[rr, cc] = self.BLACK
                        extinguished_cells += 1

        return extinguished_cells

    def move_drones(self, actions):
        extinguish_count = 0

        for drone in self.drone_names:
            discrete_action = int(actions[drone])
            vec_action = self.ACTION_LIBRARY[discrete_action]

            self.drone_states[drone] = np.clip(
                self.drone_states[drone] + vec_action,
                0,
                GRID_SIZE - 1,
            )

            if actions[drone] == 9:
                extinguish_count += self.extinguish_fire(drone)

        return extinguish_count

    def step(self, actions):
        extinguish_count = self.move_drones(actions)
        self.fire_state = self.fire_controller(self.fire_state).astype(np.int8)
        self.t += 1

        if np.all(self.fire_state != self.GREEN):
            done = True
        else:
            done = False

        return self.state(), extinguish_count, done, None

    def image(self):
        return self.COLORS[self.fire_state]


def render_fire_with_drones(env, save_path="test.png"):
    img = env.image().copy()

    # Overlay drones in blue
    for drone_name, pos in env.drone_states.items():
        r, c = pos
        img[r, c] = np.array([0, 0, 255], dtype=np.uint8)

        # Optional: make drone easier to see with a small 3x3 blue block
        r0, r1 = max(0, r - 1), min(GRID_SIZE, r + 2)
        c0, c1 = max(0, c - 1), min(GRID_SIZE, c + 2)
        img[r0:r1, c0:c1] = np.array([0, 0, 255], dtype=np.uint8)

    plt.imsave(save_path, img)


if __name__ == "__main__":
    env = DroneFireEnv(
        drone_names=["drone_0", "drone_1", "drone_2"],
        drone_start_area=[30, 30],
        fire_controller=probabilistic_fire_controller,
        drone_start_noise=5,
        ignition_points=[[30, 30], [30, 35], [32, 38]],
    )

    # Optional one-step movement test
    actions = {
        "drone_0": 0,  # stay
        "drone_1": 0,  # right
        "drone_2": 0,  # down-right
    }

    for i in range(30):
        env.step(actions)

    render_fire_with_drones(env, "test1.png")

    actions = {
        "drone_0": 9,  # stay
        "drone_1": 9,  # right
        "drone_2": 9,  # down-right
    }

    env.step(actions)

    render_fire_with_drones(env, "test2.png")

    

    print("Saved test.png")