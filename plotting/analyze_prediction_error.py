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


def plot_prediction_errors(errors_by_step, save_path: str = PLOT_SAVE_PATH):
    '''
    plot the per-step mean sampled-prediction error (+/- stdev) for each
    agent, stacked in one subplot per agent

    inputs
    ------
    errors_by_step:dict
        output of belief_model_analysis: step -> {agent: array of sample errors}
    save_path:str
        where to save the resulting figure
    '''
    agents = sorted({agent for step_errors in errors_by_step.values() for agent in step_errors})
    if not agents:
        raise ValueError('errors_by_step has no sampled_predictions - regenerate the eval results')

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
        ax.set_ylabel('error')
        ax.set_title(agent)
        ax.legend()

    axes[-1].set_xlabel('step')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    print(f'saved plot to {save_path}')


if __name__ == '__main__':
    errors_by_step = belief_model_analysis()
    plot_prediction_errors(errors_by_step)
