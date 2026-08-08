import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from util.util import load_config
from envs.marl.make_env import make_marl_env
from controllers.predator_prey_control import compute_slot_actions

DEFAULT_CONFIG = 'confs/predator_prey/3a_game.yaml'
DEFAULT_SAVE_PATH = '/home/cameron/tells/logs/marl/predator_prey_controller_decomposed_rewards.npz'


def _episode_done(terminations: dict, truncations: dict) -> bool:
    return all(bool(v) for v in terminations.values()) or all(bool(v) for v in truncations.values())


def collect_decomposed_rewards(cfg: dict, episodes: int) -> list[tuple]:
    '''
    run `episodes` controller-driven predator-prey simulations (same
    expert-controller rollout as learn/marl/train.py::_collect_controller_dataset)
    and record every agent's decomposed reward at every timestep

    the decomposed reward at step t is computed from the obs the controller
    acted on (pre-step obs), paired with the joint reward that action
    produced - mirroring how _collect_controller_dataset pairs (obs, action, reward)

    steps where the episode ends with a success bonus or an out-of-bounds
    penalty are skipped, same as plotting/predator_prey_evaluate.py's
    decomposed_reward_by_step: that terminal bonus/penalty is a flat
    team-wide add-on to the joint reward that force_share can't
    meaningfully attribute to individual agents

    returns
    -------
    list of (episode, step, agent, decomposed_reward, joint_reward) rows
    '''
    learned_agents = cfg['env']['learned_agent_list']
    rows = []

    for episode_idx in range(episodes):
        env = make_marl_env(cfg, seed=int(cfg['seed']) + episode_idx, wrap=None)
        obs, _infos = env.reset()

        for step in range(cfg['env']['max_episode_length']):
            expert_actions = compute_slot_actions(obs, env.unwrapped.obs_map)
            step_obs = obs

            action_dict = {'target': 0, **expert_actions}
            obs, rewards, terminations, truncations, infos = env.step(action_dict)
            joint_reward = float(rewards[learned_agents[0]])

            common_infos = infos['__common__']
            terminal_bonus_or_penalty = any(
                common_infos.get(agent, {}).get('success') or common_infos.get(agent, {}).get('oob')
                for agent in learned_agents
            )
            if not terminal_bonus_or_penalty:
                decomposed_reward = env.unwrapped.decompose_reward(joint_reward, step_obs)
                for agent, value in decomposed_reward.items():
                    rows.append((episode_idx, step, agent, float(value), joint_reward))

            if _episode_done(terminations, truncations):
                break

        env.close()

    return rows


def save_rows(rows: list[tuple], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        episode=np.array([row[0] for row in rows], dtype=np.int32),
        step=np.array([row[1] for row in rows], dtype=np.int32),
        agent=np.array([row[2] for row in rows]),
        decomposed_reward=np.array([row[3] for row in rows], dtype=np.float32),
        joint_reward=np.array([row[4] for row in rows], dtype=np.float32),
    )
    print(f'saved {len(rows)} rows ({rows[-1][0] + 1} episodes) to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--save-path', type=str, default=DEFAULT_SAVE_PATH)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = collect_decomposed_rewards(cfg, episodes=args.episodes)
    save_rows(rows, args.save_path)
