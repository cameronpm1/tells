import argparse
import glob
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


RESULTS_DIR = '/home/cameron/tells/logs/marl/pp_bc_noisem/checkpoint10000/results'
PLOT_SAVE_PATH = '/home/cameron/tells/test.png'


def permutation_invariant_error(pred, target):
    '''
    pred:   (N, 4)
    target: (4,) or (N, 4)

    returns
    -------
    per-sample minimum assignment distance, shape (N,)
    '''
    pred = np.asarray(pred).reshape(-1, 2, 2)
    target = np.asarray(target).reshape(-1, 2, 2)

    direct = (
        np.linalg.norm(pred[:, 0] - target[:, 0], axis=1) +
        np.linalg.norm(pred[:, 1] - target[:, 1], axis=1)
    )
    swapped = (
        np.linalg.norm(pred[:, 0] - target[:, 1], axis=1) +
        np.linalg.norm(pred[:, 1] - target[:, 0], axis=1)
    )

    return np.minimum(direct, swapped)


def belief_model_analysis(results_dir: str = RESULTS_DIR, episode: int = 1):
    '''
    load an eval results npz file and, for each step, compute the
    permutation invariant error between each of the 40 sampled belief
    predictions and the ground truth 'team' slice of that agent's obs

    inputs
    ------
    results_dir:str
        directory containing the per-episode <idx>.npz result files
    episode:int
        which episode file to load (loads '<episode>.npz')

    returns
    -------
    dict mapping step (int) -> {agent: array of 40 sample errors}
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)

    obs_map = data['obs_map'].item()

    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)

    errors_by_step = {}
    for step in steps:
        _obs, _rewards, _terminations, _truncations, infos = data[step]
        common = infos['__common__']
        sampled_predictions = common.get('sampled_predictions')
        if sampled_predictions is None:
            continue

        obs_no_noise = common['obs_no_noise']
        step_errors = {}
        for agent, samples in sampled_predictions.items():
            if torch.is_tensor(samples):
                samples = samples.detach().cpu().numpy()
            target = obs_no_noise[agent][obs_map['team']]
            step_errors[agent] = permutation_invariant_error(samples, target)

        errors_by_step[int(step)] = step_errors

    print(f'loaded {len(steps)} steps from {npz_path}')
    print(f'{len(errors_by_step)} steps have sampled_predictions')
    if errors_by_step:
        first_step = next(iter(errors_by_step))
        for agent, errs in errors_by_step[first_step].items():
            print(f'  {agent}: {errs.shape} errors, mean={errs.mean():.4f}')

    return errors_by_step


def cross_agent_estimation_error(results_dir: str = RESULTS_DIR, episode: int = 1):
    '''
    load an eval results npz file and, for each step, compute the error
    between every OTHER agent's estimate of an agent's position (the
    'team' slice of that other agent's observation) and the agent's true
    position

    for x agents there are x-1 estimates of each agent, one from every
    other agent's point of view

    inputs
    ------
    results_dir:str
        directory containing the per-episode <idx>.npz result files
    episode:int
        which episode file to load (loads '<episode>.npz')

    returns
    -------
    dict mapping step (int) -> {agent: array of (n_agents - 1) position errors}
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)

    obs_map = data['obs_map'].item()
    team_slice = obs_map['team']
    self_pos_slice = obs_map['self_pos']

    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)

    errors_by_step = {}
    for step in steps:
        obs, _rewards, _terminations, _truncations, infos = data[step]
        obs_no_noise = infos['__common__'].get('obs_no_noise')
        if not obs_no_noise:
            continue

        agents = list(obs_no_noise.keys())
        if len(agents) < 2:
            continue
        dim = (team_slice.stop - team_slice.start) // (len(agents) - 1)
        true_pos = {agent: np.asarray(obs_no_noise[agent][self_pos_slice]) for agent in agents}

        step_errors = {agent: [] for agent in agents}
        for observer in agents:
            others = [agent for agent in agents if agent != observer]
            estimate = np.asarray(obs[observer][team_slice]).reshape(-1, dim)
            for idx, target in enumerate(others):
                estimated_pos = true_pos[observer] + estimate[idx]
                step_errors[target].append(np.linalg.norm(estimated_pos - true_pos[target]))

        errors_by_step[int(step)] = {agent: np.array(errs) for agent, errs in step_errors.items()}

    print(f'loaded {len(steps)} steps from {npz_path}')
    print(f'{len(errors_by_step)} steps have cross-agent estimation errors')
    if errors_by_step:
        first_step = next(iter(errors_by_step))
        for agent, errs in errors_by_step[first_step].items():
            print(f'  {agent}: {errs.shape} estimates, mean={errs.mean():.4f}')

    return errors_by_step


def average_cross_agent_estimation_error(results_dir: str = RESULTS_DIR, episodes=range(10)):
    '''
    run cross_agent_estimation_error over multiple episodes and merge the
    results into a single errors_by_step dict, concatenating each agent's
    per-step errors across all episodes that reached that step

    inputs
    ------
    results_dir:str
        directory containing the per-episode <idx>.npz result files
    episodes:iterable of int
        which episode files to load and average over

    returns
    -------
    dict mapping step (int) -> {agent: array of pooled errors across episodes}
    '''
    combined = {}
    for episode in episodes:
        errors_by_step = cross_agent_estimation_error(results_dir=results_dir, episode=episode)
        for step, step_errors in errors_by_step.items():
            bucket = combined.setdefault(step, {})
            for agent, errs in step_errors.items():
                bucket.setdefault(agent, []).append(errs)

    return {
        step: {agent: np.concatenate(errs_list) for agent, errs_list in step_errors.items()}
        for step, step_errors in combined.items()
    }


def plot_agent_errors(errors_by_step, save_path: str = PLOT_SAVE_PATH, ylabel: str = 'error'):
    '''
    plot the per-step mean error (+/- stdev) for each agent, stacked in
    one subplot per agent

    inputs
    ------
    errors_by_step:dict
        step -> {agent: array of per-agent errors}, as returned by
        belief_model_analysis or cross_agent_estimation_error
    save_path:str
        where to save the resulting figure
    ylabel:str
        y-axis label for each subplot
    '''
    agents = sorted({agent for step_errors in errors_by_step.values() for agent in step_errors})
    if not agents:
        raise ValueError('errors_by_step is empty - regenerate the eval results')

    sns.set_theme(style='darkgrid')
    fig, axes = plt.subplots(len(agents), 1, figsize=(10, 3 * len(agents)), sharex=True)
    if len(agents) == 1:
        axes = [axes]

    for ax, agent in zip(axes, agents):
        steps = sorted(s for s, step_errors in errors_by_step.items() if agent in step_errors)
        means = np.array([errors_by_step[s][agent].mean() for s in steps])
        stds = np.array([errors_by_step[s][agent].std() for s in steps])

        sns.lineplot(x=steps, y=means, ax=ax, label='mean error')
        ax.fill_between(steps, means - stds, means + stds, alpha=0.3, label='+/- 1 stdev')
        ax.set_ylabel(ylabel)
        ax.set_title(agent)
        ax.legend()

    axes[-1].set_xlabel('step')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    print(f'saved plot to {save_path}')


def list_episodes(results_dir: str):
    '''
    return the sorted episode indices for the <idx>.npz result files
    found directly inside results_dir
    '''
    paths = glob.glob(os.path.join(results_dir, '*.npz'))
    return sorted(int(os.path.splitext(os.path.basename(p))[0]) for p in paths)


def plot_overall_average_error(errors_by_step, save_path: str, ylabel: str = 'error'):
    '''
    plot the per-step mean error (+/- stdev) pooled across every agent,
    as a single curve rather than one subplot per agent

    inputs
    ------
    errors_by_step:dict
        step -> {agent: array of per-agent errors}
    save_path:str
        where to save the resulting figure
    ylabel:str
        y-axis label for the plot
    '''
    steps = sorted(errors_by_step.keys())
    if not steps:
        raise ValueError('errors_by_step is empty - regenerate the eval results')

    means = np.array([np.concatenate(list(errors_by_step[s].values())).mean() for s in steps])
    stds = np.array([np.concatenate(list(errors_by_step[s].values())).std() for s in steps])

    sns.set_theme(style='darkgrid')
    fig, ax = plt.subplots(figsize=(10, 4))

    sns.lineplot(x=steps, y=means, ax=ax, label='mean error')
    ax.fill_between(steps, means - stds, means + stds, alpha=0.3, label='+/- 1 stdev')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('step')
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    print(f'saved plot to {save_path}')


if __name__ == '__main__':
    # python plotting/analyze_prediction_error.py --results /home/cameron/tells/logs/marl/predator_prey_fully_obs/checkpoint1250/results
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default=RESULTS_DIR,
                         help='directory containing the per-episode <idx>.npz result files')
    args = parser.parse_args()

    results_dir = args.results
    images_dir = os.path.join(results_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    episodes = list_episodes(results_dir)

    for episode in episodes:
        run_errors = cross_agent_estimation_error(results_dir=results_dir, episode=episode)
        plot_agent_errors(
            run_errors,
            save_path=os.path.join(images_dir, f'run_{episode}_error.png'),
            ylabel='position estimation error',
        )

    averaged_by_agent = average_cross_agent_estimation_error(results_dir=results_dir, episodes=episodes)
    plot_agent_errors(
        averaged_by_agent,
        save_path=os.path.join(images_dir, 'average_by_agent.png'),
        ylabel=f'avg error ({len(episodes)} runs)',
    )

    plot_overall_average_error(
        averaged_by_agent,
        save_path=os.path.join(images_dir, 'average_overall.png'),
        ylabel=f'avg error ({len(episodes)} runs, all agents)',
    )
