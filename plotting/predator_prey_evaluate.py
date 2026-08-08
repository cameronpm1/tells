import argparse
import glob
import os
import sys
from types import SimpleNamespace

import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.marl.predator_prey_env import PredatorPreyEnv
from overlay_plot_predator_prey import load_episode_positions, load_episode_beliefs


RESULTS_DIR = '/home/cameron/tells/logs/marl/predator_prey_fully_obs/checkpoint1000/results'
RESULTS_PF_DIR = '/home/cameron/tells/logs/marl/predator_prey_fully_obs/checkpoint1000/results_pf'
BELIEF_ERROR_SAVE_PATH = '/home/cameron/tells/test_belief_error.png'
DECOMPOSED_REWARD_SAVE_PATH = '/home/cameron/tells/test_decomposed_reward.png'

# mean decomposed reward over 500 controller-driven episodes
# (evals/marl/collect_decomposed_rewards.py), with terminal success/oob
# steps filtered out the same way decomposed_reward_by_step does below
CONTROLLER_MEAN_DECOMPOSED_REWARD = 0.448082


def list_episodes(results_dir: str):
    '''
    return the sorted episode indices for the <idx>.npz result files
    found directly inside results_dir
    '''
    paths = glob.glob(os.path.join(results_dir, '*.npz'))
    return sorted(int(os.path.splitext(os.path.basename(p))[0]) for p in paths)


def load_controller_cfg(results_dir: str) -> dict:
    '''
    walk up from a checkpoint's results dir (<run_dir>/<checkpoint>/results)
    to the run's config.yaml and pull out env.controller_kwargs - needed to
    recompute the adversary force each predator applies to the prey, the
    same way PredatorPreyEnv.decompose_reward does
    '''
    checkpoint_dir = os.path.dirname(os.path.normpath(results_dir))
    run_dir = os.path.dirname(checkpoint_dir)
    config_path = os.path.join(run_dir, 'config.yaml')

    with open(config_path, 'r') as stream:
        cfg = yaml.safe_load(stream)

    return cfg['env']['controller_kwargs']


def compute_decomposed_reward(obs_no_noise: dict, rewards: dict, obs_map: dict, controller_cfg: dict) -> dict:
    '''
    recompute the per-agent decomposed reward for a single step by calling
    the real PredatorPreyEnv.decompose_reward (envs/marl/predator_prey_env.py)
    directly, rather than reimplementing its logic here - decompose_reward
    only touches self.agents / self.controller_cfg / self.obs_map, so a
    lightweight stub standing in for the env is enough to call it unbound

    inputs
    ------
    obs_no_noise:dict
        agent -> ground-truth observation for that step (infos['__common__']['obs_no_noise'])
    rewards:dict
        agent -> reward for that step
    obs_map:dict
        slices dict loaded from the eval npz ('obs_map')
    controller_cfg:dict
        env.controller_kwargs from the run's config.yaml

    returns
    -------
    dict: agent -> decomposed reward
    '''
    env_stub = SimpleNamespace(
        agents=sorted(obs_no_noise.keys()),
        controller_cfg=controller_cfg,
        obs_map=obs_map,
    )
    joint_reward = float(np.mean(list(rewards.values())))

    return PredatorPreyEnv.decompose_reward(env_stub, joint_reward, obs_no_noise)


def belief_error_by_step(results_dir: str, episode: int) -> dict:
    '''
    load a single episode's eval results and, for each step, compute each
    predator's individual belief error the same way overlay_plot_predator_prey
    does: reconstruct absolute ground-truth predator positions
    (load_episode_positions) and each observer's believed absolute position
    of its teammates (load_episode_beliefs), then take the position error
    between them for every observer/teammate pair and average across pairs

    this differs from infos['__common__']['belief_error'] (a
    permutation-invariant error the env computes internally) by matching
    teammates to their known identity instead

    returns
    -------
    dict: step (int) -> belief error averaged over agent pairs (float)
    '''
    predator_positions, _target_positions, _true_target_positions, _goal_position = load_episode_positions(
        results_dir=results_dir, episode=episode,
    )
    beliefs = load_episode_beliefs(results_dir=results_dir, episode=episode)
    predator_names = sorted(predator_positions)
    n_steps = predator_positions[predator_names[0]].shape[0]

    errors_by_step = {}
    for t in range(n_steps):
        pair_errors = [
            np.linalg.norm(predator_positions[target][t] - beliefs[observer][target][t])
            for observer in predator_names
            for target in predator_names
            if target != observer
        ]
        errors_by_step[t + 1] = float(np.mean(pair_errors))

    return errors_by_step


def decomposed_reward_by_step(results_dir: str, episode: int, controller_cfg: dict) -> dict:
    '''
    load a single episode's eval results and, for each step, recompute the
    per-agent decomposed reward and average it across agents

    steps where the episode terminated with a success bonus or an
    out-of-bounds penalty are skipped: decompose_reward apportions the
    joint reward by each predator's adversary-force contribution, which
    only makes sense for the smooth per-step reward - a terminal
    success/oob bonus is a flat team-wide add-on that force_share can't
    meaningfully attribute to individual agents

    returns
    -------
    dict: step (int) -> decomposed reward averaged over agents (float)
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)

    obs_map = data['obs_map'].item()
    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)

    reward_by_step = {}
    for step in steps:
        _obs, rewards, _terminations, _truncations, infos = data[step]
        common_infos = infos['__common__']
        obs_no_noise = common_infos.get('obs_no_noise')
        if not obs_no_noise:
            continue

        terminal_bonus_or_penalty = any(
            common_infos.get(agent, {}).get('success') or common_infos.get(agent, {}).get('oob')
            for agent in obs_no_noise
        )
        if terminal_bonus_or_penalty:
            continue

        decomposed_reward = compute_decomposed_reward(obs_no_noise, rewards, obs_map, controller_cfg)
        reward_by_step[int(step)] = float(np.mean(list(decomposed_reward.values())))

    return reward_by_step


def average_over_episodes(per_episode_fn, results_dir: str, episodes) -> dict:
    '''
    run per_episode_fn(results_dir, episode) over every episode and pool
    the results by step

    returns
    -------
    dict: step (int) -> array of per-episode values at that step
    '''
    combined = {}
    for episode in episodes:
        values_by_step = per_episode_fn(results_dir, episode)
        for step, value in values_by_step.items():
            combined.setdefault(step, []).append(value)

    return {step: np.array(values) for step, values in combined.items()}


def plot_average_with_std(
    series: dict,
    save_path: str,
    ylabel: str,
    title: str,
    reference_line: float = None,
    reference_label: str = None,
):
    '''
    plot the per-step mean (+/- 1 stdev, pooled across episodes) for one or
    more named series, each as its own curve with a shaded std-dev band

    inputs
    ------
    series:dict
        label -> values_by_step (step -> array of per-episode values, as
        returned by average_over_episodes)
    save_path:str
        where to save the resulting figure
    ylabel:str
        y-axis label
    title:str
        plot title
    reference_line:float
        if given, draw a black dotted horizontal line at this value (e.g.
        CONTROLLER_MEAN_DECOMPOSED_REWARD)
    reference_label:str
        legend label for the reference line
    '''
    sns.set_theme(style='darkgrid')
    fig, ax = plt.subplots(figsize=(10, 4))

    for label, values_by_step in series.items():
        steps = sorted(values_by_step.keys())
        if not steps:
            raise ValueError(f'values_by_step for "{label}" is empty - regenerate the eval results')

        means = np.array([values_by_step[s].mean() for s in steps])
        stds = np.array([values_by_step[s].std() for s in steps])

        line = sns.lineplot(x=steps, y=means, ax=ax, label=f'{label} mean')
        color = line.get_lines()[-1].get_color()
        ax.fill_between(steps, means - stds, means + stds, alpha=0.2, color=color, label=f'{label} +/- 1 stdev')

    if reference_line is not None:
        ax.axhline(reference_line, color='black', linestyle=':', label=reference_label)

    ax.set_ylabel(ylabel)
    ax.set_xlabel('step')
    ax.set_title(title, fontweight='bold')
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    print(f'saved plot to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default=RESULTS_DIR,
                         help='directory containing the allocentric per-episode <idx>.npz result files')
    parser.add_argument('--results-pf', type=str, default=RESULTS_PF_DIR,
                         help='directory containing the particle-filter per-episode <idx>.npz result files')
    parser.add_argument('--belief-error-save-path', type=str, default=BELIEF_ERROR_SAVE_PATH,
                         help='where to save the belief error plot')
    parser.add_argument('--decomposed-reward-save-path', type=str, default=DECOMPOSED_REWARD_SAVE_PATH,
                         help='where to save the decomposed reward plot')
    args = parser.parse_args()

    result_dirs = {'Allocentric': args.results, 'PF': args.results_pf}

    belief_error = {}
    decomposed_reward = {}
    episode_counts = {}

    for label, results_dir in result_dirs.items():
        episodes = list_episodes(results_dir)
        episode_counts[label] = len(episodes)
        controller_cfg = load_controller_cfg(results_dir)

        belief_error[label] = average_over_episodes(belief_error_by_step, results_dir, episodes)
        decomposed_reward[label] = average_over_episodes(
            lambda rdir, ep: decomposed_reward_by_step(rdir, ep, controller_cfg),
            results_dir,
            episodes,
        )

    run_counts = ', '.join(f'{label}: {n}' for label, n in episode_counts.items())

    plot_average_with_std(
        belief_error,
        save_path=args.belief_error_save_path,
        ylabel=f'belief error ({run_counts})',
        title='Average Belief Error over Episode Steps',
    )

    plot_average_with_std(
        decomposed_reward,
        save_path=args.decomposed_reward_save_path,
        ylabel=f'decomposed reward ({run_counts})',
        title='Average Decomposed Reward over Episode Steps',
        reference_line=CONTROLLER_MEAN_DECOMPOSED_REWARD,
        reference_label=f'controller mean',
    )
