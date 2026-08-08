import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from util.util import load_config
from envs.marl.make_env import make_marl_env
from controllers.football_control import compute_rondo_actions

DEFAULT_CONFIG = 'confs/football/5a_game.yaml'
DEFAULT_SAVE_PATH = '/home/cameron/tells/logs/marl/football_controller_decomposed_rewards.npz'


def _episode_done(terminations: dict, truncations: dict) -> bool:
    return all(bool(v) for v in terminations.values()) or all(bool(v) for v in truncations.values())


def collect_decomposed_rewards(cfg: dict, episodes: int) -> list[tuple]:
    '''
    run `episodes` controller-driven football (rondo) simulations, driven by
    compute_rondo_actions the same way learn/marl/train.py::_collect_controller_dataset
    does, and record every agent's decomposed reward at every timestep via
    CirclePass5v1Env.decompose_reward (envs/marl/football_env.py)

    steps where the episode ends with an out-of-bounds or target-steal
    penalty are skipped, same rationale as evals/marl/collect_decomposed_rewards.py
    (predator_prey): that terminal penalty is a flat team-wide add-on to the
    joint reward that the pass/openness contribution scale can't
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
            actions = compute_rondo_actions(obs, env.unwrapped.obs_map)
            last_obs = obs

            obs, rewards, terminations, truncations, infos = env.step(actions)
            joint_reward = float(rewards[learned_agents[0]])

            common_infos = infos['__common__']
            target_has_ball = bool(obs['target'][env.unwrapped.obs_map['target_ball_owned']][0])
            terminal_penalty = target_has_ball or any(
                bool(common_infos.get(agent, {}).get('oob'))
                for agent in learned_agents
            )
            if not terminal_penalty:
                decomposed_reward = env.unwrapped.decompose_reward(joint_reward, obs, last_obs)
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

    decomposed_reward = np.array([row[3] for row in rows], dtype=np.float32)
    print(
        'decomposed reward: mean %.6f std %.6f min %.6f max %.6f'
        % (decomposed_reward.mean(), decomposed_reward.std(), decomposed_reward.min(), decomposed_reward.max())
    )
