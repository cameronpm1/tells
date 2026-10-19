import numpy as np
from copy import deepcopy
from typing import Optional

from mpe2._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
from controllers.predator_prey_swarm_control import compute_slot_actions
from envs.marl.predator_prey_env import GRID_SIZE, PredatorPreyEnv, ScenarioEnv, PredatorPreyScenario


def swarm_summary(rel_positions: np.ndarray, eps: float = 1e-2):
    '''
    covariance ellipse of the swarm as (mu_x, mu_y, log size, elongation_1, elongation_2)
    '''
    mu = rel_positions.mean(axis=0)
    diff = rel_positions - mu
    cov = diff.T @ diff / len(rel_positions)
    trace = cov[0, 0] + cov[1, 1]
    elongation = np.array([cov[0, 0] - cov[1, 1], 2 * cov[0, 1]]) / (trace + eps)
    return np.concatenate((mu, [np.log(trace + eps)], elongation))

class PredatorPreySwarmEnv(PredatorPreyEnv):

    '''
    predator prey env with a variable size predator swarm, teammates observed through swarm shape
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.obs_map = {
            'self': slice(0, 4),
            'target': slice(4, 8),
            'team': slice(8, 13),
            'target_obs': slice(8, 8 + 2 * len(self.agents)),
            'self_pos': slice(2, 4),
            'target_goal': slice(4, 6),
            'target_pos': slice(6, 8),
            'swarm_pos': slice(8, 10),
            'state_space': (13,)
        }

    def behavior_cloning_reward(self, action_dict: dict) -> dict:
        controller_actions = compute_slot_actions(deepcopy(self.obs), self.obs_map)
        action_matches = [
            float(int(action_dict[agent]) == self.boundary_safe_action(agent, int(controller_actions[agent])))
            for agent in self.agents if agent in action_dict and agent in controller_actions
        ]
        controller_action_match = float(np.mean(action_matches)) if action_matches else 0.0

        return {
            'controller_action_error': 1.0 - controller_action_match,
            'controller_action_match': controller_action_match,
            'controller_action_reward': controller_action_match,
        }

    def compute_team_reward(self, metrics: dict):
        team_reward = (
            + (self.reward_cfg['bc_scale'] * metrics['controller_action_reward'])
            + (self.reward_cfg['coverage_scale'] * metrics['coverage_score'])
            - self.reward_cfg['step_cost']
        )

        if metrics['hold']:
            team_reward += self.reward_cfg['hold_scale']

        if metrics['success']:
            team_reward += self.reward_cfg['success_bonus']

        return team_reward

    def team_error(
        self,
        estimate: np.ndarray,
        agent: str
    ):
        return float(np.linalg.norm(estimate[self.obs_map['team']] - self.obs[agent][self.obs_map['team']]))


class SwarmScenarioEnv(ScenarioEnv):

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

        scenario = PredatorPreySwarmScenario(**scenario_kwargs)
        world = scenario.make_world()

        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
            dynamic_rescaling=True,
        )

class PredatorPreySwarmScenario(PredatorPreyScenario):

    def __init__(self, swarm_eps: float = 1e-2, **kwargs):
        super().__init__(**kwargs)
        self.swarm_eps = swarm_eps

    def observation(self, agent, world):

        pos = agent.state.p_pos
        goal = world.landmarks[0].state.p_pos - pos
        target = [other for other in world.agents if other.adversary][0]

        obs_self = np.concatenate((agent.state.p_vel, pos))
        obs_target = np.concatenate((goal, target.state.p_pos - pos))

        rel_positions = np.array([
            other.state.p_pos - pos for other in world.agents
            if not other.adversary and other is not agent
        ])

        if agent.adversary:
            return np.concatenate((obs_self, obs_target, rel_positions.flatten()))

        return np.concatenate((obs_self, obs_target, swarm_summary(rel_positions, self.swarm_eps)))


env = make_env(SwarmScenarioEnv)
parallel_env = parallel_wrapper_fn(env)
