
import sys
import gym
import math
import numpy as np
from gym import spaces
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
SCENARIO_DIR = THIS_FILE.parent
PROJECT_ROOT = THIS_FILE.parents[2]
FOOTBALL_REPO = PROJECT_ROOT / "external" / "football"

if str(FOOTBALL_REPO) not in sys.path:
    sys.path.insert(0, str(FOOTBALL_REPO))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gfootball.scenarios import *
import gfootball.env as football_env

PROJECT_ROOT = Path(__file__).resolve()
FOOTBALL_REPO = PROJECT_ROOT / "external" / "football"

sys.path.insert(0, str(FOOTBALL_REPO))


def register_file_as_grf_scenario(scenario_name: str):
    """
    Registers this current Python module as:
        gfootball.scenarios.<scenario_name>

    This allows GRF to find build_scenario() even though the scenario is
    defined in the same file as the environment wrapper.
    """

    module_name = f"gfootball.scenarios.{scenario_name}"
    current_module = sys.modules[__name__]

    if not hasattr(current_module, "build_scenario"):
        raise RuntimeError(
            f"{__name__} does not define build_scenario(). "
            "Make sure build_scenario() is defined before creating the env."
        )

    sys.modules[module_name] = current_module

class CirclePass5v1Env(gym.Env):
    """
    Multi-agent 5v1 rondo-style drill.

    Agents:
        left_0 ... left_4: passing team
        right_0: middle defender

    Objective:
        left team keeps possession and passes around the circle.
        right defender tries to steal the ball.
    """

    def __init__(self, render=False):
        print("[CirclePass5v1Env] __init__ start", flush=True)
        super().__init__()

        self.n_left = 6
        self.n_right = 2
        self.n_agents = self.n_left + self.n_right

        scenario_name = "academy_circle_pass_5v1"

        print("[CirclePass5v1Env] registering scenario", flush=True)
        register_file_as_grf_scenario(scenario_name)

        print("[CirclePass5v1Env] creating GRF environment", flush=True)
        self.env = football_env.create_environment(
            env_name=scenario_name,
            representation="raw",
            rewards="scoring",
            render=render,
            number_of_left_players_agent_controls=self.n_left,
            number_of_right_players_agent_controls=self.n_right,
        )

        print("[CirclePass5v1Env] GRF environment created", flush=True)

        self.action_space = spaces.MultiDiscrete([19] * self.n_agents)

        self.observation_space = spaces.Tuple([
            spaces.Dict({}) for _ in range(self.n_agents)
        ])

        self.prev_ball_owned_team = None
        self.prev_ball_owned_player = None
        self.prev_left_possession_player = None
        self.successful_passes = 0

        print("[CirclePass5v1Env] __init__ done", flush=True)

    def reset(self):
        obs = self.env.reset()

        self.prev_ball_owned_team = None
        self.prev_ball_owned_player = None
        self.prev_left_possession_player = None
        self.successful_passes = 0

        return obs

    def step(self, actions):
        if len(actions) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} actions, got {len(actions)}")

        actions = [int(a) for a in actions]

        print("[CirclePass5v1Env] before self.env.step", actions, flush=True)
        actions[0:2] = [0,0]
        obs, base_reward, done, info = self.env.step(actions)
        print("[CirclePass5v1Env] after self.env.step", flush=True)

        shaped_rewards, drill_done, drill_info = self._compute_drill_rewards(obs)

        done = bool(done or drill_done)
        info.update(drill_info)

        return obs, shaped_rewards, done, info

    def nearest_left_passer_to_ball(self, state):
        ball_xy = np.asarray(state["ball"][:2], dtype=np.float32)
        left_team_all = np.asarray(state["left_team"], dtype=np.float32)

        # With non-controllable GK at index 0, passers are indices 1..5.
        left_passers = left_team_all[1:1 + self.n_left]

        dists = np.linalg.norm(left_passers - ball_xy, axis=1)
        local_idx = int(np.argmin(dists))
        grf_player_idx = local_idx + 1

        return grf_player_idx, float(dists[local_idx])

    def _compute_drill_rewards(self, obs):
        """
        obs is a list of raw observations, one per controlled player.
        Use obs[0] as the global state reference because raw state includes both teams.
        """

        state = obs[0]

        ball_owned_team = state["ball_owned_team"]      # -1 none, 0 left, 1 right
        ball_owned_player = state["ball_owned_player"]  # player index within owning team
        ball = np.array(state["ball"][:2], dtype=np.float32)

        left_team = np.array(state["left_team"], dtype=np.float32)
        right_team = np.array(state["right_team"], dtype=np.float32)

        rewards = np.zeros(self.n_agents, dtype=np.float32)
        done = False

        # Possession shaping.
        if ball_owned_team == 0:
            rewards[:self.n_left] += 0.02
            rewards[self.n_left:] -= 0.02

        if ball_owned_team == 1:
            rewards[:self.n_left] -= 1.0
            rewards[self.n_left:] += 1.0
            done = True

        # Successful pass detection:
        # ball remains with left team, but owner changed from one passer to another.
        successful_pass = (
            self.prev_ball_owned_team == 0
            and ball_owned_team == 0
            and self.prev_ball_owned_player is not None
            and ball_owned_player != self.prev_ball_owned_player
            and 0 <= ball_owned_player < self.n_left
        )

        if successful_pass:
            self.successful_passes += 1
            rewards[:self.n_left] += 0.25
            rewards[self.n_left:] -= 0.10

        # Reward passers for keeping circular spacing.
        center = np.array([0.0, 0.0], dtype=np.float32)
        desired_radius = np.array([0.28, 0.18], dtype=np.float32)

        for i in range(self.n_left):
            normalized = (left_team[i] - center) / desired_radius
            radius_error = abs(np.linalg.norm(normalized) - 1.0)
            rewards[i] -= 0.01 * radius_error

        # Reward defender for closing distance to ball.
        defender_pos = right_team[0]
        defender_ball_dist = np.linalg.norm(defender_pos - ball)
        rewards[self.n_left] += 0.01 * (1.0 - defender_ball_dist)

        # Optional curriculum target, episode ends after 10 completed passes.
        if self.successful_passes >= 10:
            rewards[:self.n_left] += 1.0
            rewards[self.n_left:] -= 1.0
            done = True

        self.prev_ball_owned_team = ball_owned_team
        self.prev_ball_owned_player = ball_owned_player

        info = {
            "successful_passes": self.successful_passes,
            "ball_owned_team": ball_owned_team,
            "ball_owned_player": ball_owned_player,
            "stolen_by_defender": ball_owned_team == 1,
        }

        return rewards, done, info

    def render_rgb(self):
        """
        Returns the current GRF frame as an RGB uint8 image.

        Output:
            np.ndarray with shape (H, W, 3), dtype=np.uint8
        """

        frame = self.env.render(mode="rgb_array")

        if frame is None:
            raise RuntimeError(
                "render_rgb() returned None. Create the environment with render=True."
            )

        frame = np.asarray(frame)

        if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            raise RuntimeError(f"Unexpected frame shape from renderer: {frame.shape}")

        # Drop alpha channel if present.
        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]

        return frame.astype(np.uint8)

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()

def build_scenario(builder):
    builder.config().game_duration = 600
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = False
    builder.config().end_episode_on_score = False

    center_x = 0.0
    center_y = 0.0
    rx = 0.28
    ry = 0.18
    n_passers = 5

    passer_positions = []
    for i in range(n_passers):
        theta = 2.0 * math.pi * i / n_passers
        x = center_x + rx * math.cos(theta)
        y = center_y + ry * math.sin(theta)
        passer_positions.append((x, y))

    builder.SetBallPosition(passer_positions[0][0], passer_positions[0][1])

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK, lazy=True, controllable=True)

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(1.0, 0.0, e_PlayerRole_GK, lazy=True, controllable=True)

    builder.SetTeam(Team.e_Left)
    for x, y in passer_positions:
        builder.AddPlayer(x, y, e_PlayerRole_CM, lazy=False, controllable=True)

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(center_x, center_y, e_PlayerRole_CB, lazy=False, controllable=True)