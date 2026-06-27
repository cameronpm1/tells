
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

    def __init__(
            self, 
            agents,
            render=False
        ):
        super().__init__()

        self.agents = agents
        self.n_agents = len(agents)
        self.n_left = 6
        self.n_right = 2

        self. env = self._init_scenario(scenario_name='academy_circle_pass_5v1')


        self.action_space = spaces.MultiDiscrete([19] * self.n_agents)

        self.observation_space = spaces.Tuple([
            spaces.Dict({}) for _ in range(self.n_agents)
        ])

        self.prev_ball_owned_player = -1
        self.pass_count = 0

        self.pass_reward = 1.0
        self.target_steal_penalty = -10.0
        self.target_steal_reward = 10.0

        self.obs_map = {
            {
            'self': slice(0, 4),
            'target': slice(4, 7),
            'team': slice(7, 7 + 2 * (len(self.agents) - 1)),
            'target_obs': slice(7, 7 + 2 * (len(self.agents))),
            'self_pos': slice(2, 4),
            'target_ball': slice(4, 5),
            'target_pos': slice(5, 7),
        }
        }

    def _init_scenario(self, scenario_name):
        print("[CirclePass5v1Env] registering scenario", flush=True)
        register_file_as_grf_scenario(scenario_name)

        print("[CirclePass5v1Env] creating GRF environment", flush=True)
        env = football_env.create_environment(
            env_name=scenario_name,
            representation="raw",
            rewards="scoring",
            render=render,
            number_of_left_players_agent_controls=self.n_left,
            number_of_right_players_agent_controls=self.n_right,
        )
        print("[CirclePass5v1Env] GRF environment created", flush=True)

        return env

    def reset(self):

        self.prev_ball_owned_team = -1
        self.prev_ball_owned_player = -1
        self.pass_count = 0

        obs = self.env.reset()

        return obs

    def step(self, actions):

        actions = [int(a) for a in actions]
        actions = np.concatenate(([0],actions[0:self.n_agents],[0],[actions[-1]]))

        actions[0:2] = [0,0] # set goalie actions to 0
        obs, base_reward, done, info = self.env.step(actions)

        obs_no_gk = np.delete(obs,[0,self.n_agents+1])
        new_obs = self._convert_obs(obs_no_gk)
        reward = self._compute_team_reward(new_obs)
        drill_done = False
        done = bool(done or drill_done)
        info = {}

        reward = [reward for agent in self.agents]
        done = [done for agent in self.agents]
        info = [info for agent in self.agents]

        return new_obs, shaped_rewards, done, info

    def _convert_obs(self, obs):

        new_obs = {}

        for i,agent_obs in enumerate(obs):
            if i < self.n_agents:
                agent_name = self.agents[i]
                team = agent_obs['left_team'][1:]
                pos = team[i]
                vel = agent_obs['left_team_direction'][i+1]
                team = np.delete(team,i,axis=0).flatten() - np.tile(pos,self.n_agents-1)
                if agent_obs['ball_owned_team'] == 0 and agent_obs['ball_owned_player'] == i + 1:
                    ball = 1
                else:
                    ball = 0

                target_pos = agent_obs['right_team'][1] - pos
            else:
                agent_name = 'target'
                team = agent_obs['right_team'][1:]
                pos = agent_obs['left_team'][1]
                vel = agent_obs['left_team_direction'][1]
                team = team.flatten() - np.tile(pos,self.n_agents)
                if agent_obs['ball_owned_team'] == 0 and agent_obs['ball_owned_player'] == 1:
                    ball = 1
                else:
                    ball = 0

                target_pos = agent_obs['left_team'][1] - pos

            new_obs[agent_name] = np.concatenate((vel,pos,[ball],target_pos,team))

        return new_obs


    def _compute_team_reward(self, obs):
        
        team_ball_state = [obs[agent][self.obs_map['target_ball']] for agent in self.agents]
        target_has_ball = obs['target'][self.obs_map['target_ball']]

        if target_has_ball:
            return self.target_steal_penalty
        else:
            ball_ownership = np.where(team_ball_state == 1)
            if len(ball_ownership) > 0:
                if self.previous_ball_owned_player == -1:
                    self.previous_ball_owned_player = ball_ownership[0]
                else:
                    if self.previous_ball_owned_player != ball_ownership[0]:
                        self.previous_ball_owned_player = ball_ownership[0]
                        return self.pass_reward

        return 0.0

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
    ry = 0.28
    n_passers = 5

    passer_positions = []
    for i in range(n_passers):
        theta = 2.0 * math.pi * i / n_passers
        x = center_x + rx * math.cos(theta)
        y = center_y + ry * math.sin(theta)
        passer_positions.append((x, y))

    builder.SetBallPosition(passer_positions[0][0], passer_positions[0][1])

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-10.0, 0.0, e_PlayerRole_GK, lazy=True, controllable=True)

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(10.0, 0.0, e_PlayerRole_GK, lazy=True, controllable=True)

    builder.SetTeam(Team.e_Left)
    for x, y in passer_positions:
        builder.AddPlayer(x, y, e_PlayerRole_CM, lazy=False, controllable=True)

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(center_x, center_y, e_PlayerRole_CB, lazy=False, controllable=True)