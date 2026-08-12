import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D


#RESULTS_DIR = '/home/cameron/tells/logs/marl/predator_prey_fully_obs/checkpoint1000/results'
RESULTS_DIR = '/home/cameron/tells/logs/marl/predator_prey_fully_obs/controller_results'
PLOT_SAVE_PATH = '/home/cameron/tells/test.png'
WORLD_STATE_SAVE_PATH = '/home/cameron/tells/test_world_state.png'

# matches the entity colors drawn in PredatorPreyEnv.render_rgb / ScenarioEnv.draw,
# with each predator given its own shade so overlapping trails stay legible
PREDATOR_COLORS = ['#4646aa', '#5daacf', '#8a5acf']
TARGET_COLOR = '#aa4646'
GOAL_COLOR = '#32aa32'

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'


def _fade_cmap(hex_color: str, name: str):
    '''
    build a colormap that ramps from a light tint of hex_color (t=0) up to
    the fully saturated hex_color (t=1), used to fade earlier positions in
    an episode and saturate later ones
    '''
    base = np.array(to_rgb(hex_color))
    light = 1.0 - (1.0 - base) * 0.12
    return LinearSegmentedColormap.from_list(name, [light, base])


def load_episode_positions(results_dir: str, episode: int = 1):
    '''
    load a single episode's eval results and reconstruct absolute (x, y)
    positions for every predator, the target (as seen by each predator),
    and the goal at each step

    positions are recovered from each predator's own observation, using
    obs_map['self_pos'] for the predator's own absolute position and the
    relative slices obs_map['target_pos'] / obs_map['target_goal'] (both
    stored relative to that predator) to recover target/goal position

    the target is also reconstructed separately per predator (rather than
    once, shared) because some pipelines (e.g. particle-filter belief data)
    give each agent a different observed/estimated target position - using
    one agent's reading for every panel would silently mislabel the others'

    true_target_positions comes from infos['target'] instead - the
    target's own observation, as returned directly by the env (see
    RLLibWrapper.step / PFWrapper.step: `infos['target'] = obs['target']`
    before the 'target' key is stripped from the returned obs dict) - this
    is the actual target position, not any predator's (possibly noisy or
    estimated) reading of it, so it's what the overall ground-truth panel
    should use

    inputs
    ------
    results_dir:str
        directory containing the per-episode <idx>.npz result files
    episode:int
        which episode file to load (loads '<episode>.npz')

    returns
    -------
    predator_positions:dict of agent -> (T, 2) array
    target_positions:dict of agent -> (T, 2) array, that agent's own
        observed target position
    true_target_positions:(T, 2) array, the target's own reported position
    goal_position:(2,) array
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)

    obs_map = data['obs_map'].item()
    self_pos_slice = obs_map['self_pos']
    target_pos_slice = obs_map['target_pos']
    goal_slice = obs_map['target_goal']

    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)
    if not steps:
        raise ValueError(f'no steps found in {npz_path}')

    first_obs = data[steps[0]][0]
    predator_names = sorted(first_obs.keys())

    predator_positions = {name: [] for name in predator_names}
    target_positions = {name: [] for name in predator_names}
    true_target_positions = []
    goal_position = None

    for step in steps:
        obs, _rewards, _terminations, _truncations, infos = data[step]
        for name in predator_names:
            self_pos = np.asarray(obs[name][self_pos_slice])
            predator_positions[name].append(self_pos)
            target_positions[name].append(self_pos + np.asarray(obs[name][target_pos_slice]))

        true_target_positions.append(np.asarray(infos['target'][self_pos_slice]))

        if goal_position is None:
            ref_obs = obs[predator_names[0]]
            ref_pos = np.asarray(ref_obs[self_pos_slice])
            goal_position = ref_pos + np.asarray(ref_obs[goal_slice])

    predator_positions = {name: np.stack(pos) for name, pos in predator_positions.items()}
    target_positions = {name: np.stack(pos) for name, pos in target_positions.items()}
    true_target_positions = np.stack(true_target_positions)

    return predator_positions, target_positions, true_target_positions, goal_position


def load_episode_beliefs(results_dir: str, episode: int = 1):
    '''
    load a single episode's eval results and, for each step, convert every
    agent's belief about its teammates (infos['__common__']['sampled_predictions'],
    produced by the belief model) into absolute positions

    sampled_predictions[observer] is that observer's estimate of the
    obs_map['team'] slice - the relative positions of its teammates, in the
    same order as infos['__common__']['obs_no_noise'].keys() with the
    observer itself removed (mirrors analyze_prediction_error.cross_agent_estimation_error)
    - each estimate is anchored to the observer's own ground-truth position
    (obs_no_noise) to recover an absolute estimated position for each teammate

    inputs
    ------
    results_dir:str
        directory containing the per-episode <idx>.npz result files
    episode:int
        which episode file to load (loads '<episode>.npz')

    returns
    -------
    dict: observer -> {target_agent: (T, 2) array of estimated positions}
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)

    obs_map = data['obs_map'].item()
    self_pos_slice = obs_map['self_pos']
    team_slice = obs_map['team']

    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)
    if not steps:
        raise ValueError(f'no steps found in {npz_path}')

    first_common = data[steps[0]][4]['__common__']
    agents = list(first_common['obs_no_noise'].keys())
    dim = (team_slice.stop - team_slice.start) // (len(agents) - 1)

    estimated = {observer: {target: [] for target in agents if target != observer} for observer in agents}

    for step in steps:
        _obs, _rewards, _terminations, _truncations, infos = data[step]
        common = infos['__common__']
        obs_no_noise = common['obs_no_noise']
        sampled_predictions = common['sampled_predictions']

        for observer in agents:
            others = [agent for agent in agents if agent != observer]
            observer_pos = np.asarray(obs_no_noise[observer][self_pos_slice])
            estimate = np.asarray(sampled_predictions[observer]).reshape(-1, dim)

            for idx, target in enumerate(others):
                estimated[observer][target].append(observer_pos + estimate[idx])

    return {
        observer: {target: np.stack(positions) for target, positions in targets.items()}
        for observer, targets in estimated.items()
    }


def _warn_if_window_clamped(start_idx: int, end_idx: int, total_steps: int):
    '''
    print a warning if the requested [start_idx, end_idx) window runs past
    the end of the episode - the plotted data (and title) end up clamped
    to the steps that actually exist rather than the requested range

    raises if start_idx itself is past the end of the episode, since then
    there's no data left to clamp to
    '''
    if start_idx >= total_steps:
        raise ValueError(
            f'requested start step {start_idx} is past the end of the episode '
            f'({total_steps} steps, 0-{total_steps - 1})'
        )

    if end_idx > total_steps:
        print(
            f'requested window goes up to step {end_idx - 1}, but the episode only has '
            f'{total_steps} steps (0-{total_steps - 1}); clamping to {start_idx}-{total_steps - 1}'
        )


def _draw_ground_truth(ax, predator_positions, target_positions, goal_position, fade):
    '''
    draw the faded -> saturated dot trails for every predator, the target,
    and a static goal marker onto ax; returns the legend handles
    '''
    legend_handles = []

    for idx, (name, positions) in enumerate(predator_positions.items()):
        color = PREDATOR_COLORS[idx % len(PREDATOR_COLORS)]
        cmap = _fade_cmap(color, f'predator_{idx}')

        ax.scatter(positions[:, 0], positions[:, 1], color=cmap(fade), s=35, zorder=3)
        ax.scatter(*positions[-1], color=color, s=150, zorder=4)

        legend_handles.append(Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
                                      markeredgecolor='none', markersize=9, label=name))

    target_cmap = _fade_cmap(TARGET_COLOR, 'target')
    ax.scatter(target_positions[:, 0], target_positions[:, 1], color=target_cmap(fade), s=35, zorder=3)
    ax.scatter(*target_positions[-1], color=TARGET_COLOR, s=150, zorder=4)
    legend_handles.append(Line2D([0], [0], marker='o', color='none', markerfacecolor=TARGET_COLOR,
                                  markeredgecolor='none', markersize=9, label='target'))

    ax.scatter(*goal_position, color=GOAL_COLOR, marker='o', s=150, zorder=5)
    legend_handles.append(Line2D([0], [0], marker='o', color='none', markerfacecolor=GOAL_COLOR,
                                  markeredgecolor='none', markersize=9, label='goal'))

    return legend_handles


def _set_zoom_bounds(ax, positions_list, goal_position):
    '''
    zoom ax in on goal_position, sized to fit every point in positions_list
    plus a small margin
    '''
    all_positions = np.concatenate([*positions_list, goal_position[None, :]])
    radius = np.max(np.linalg.norm(all_positions - goal_position, axis=1))
    bound = radius * 1.05 + 0.05
    ax.set_xlim(goal_position[0] - bound, goal_position[0] + bound)
    ax.set_ylim(goal_position[1] - bound, goal_position[1] + bound)
    ax.set_aspect('equal')


def _tight_bounds_and_goal_fraction(positions_list, goal_position, margin: float = 0.3):
    '''
    fit a square box tightly around every point in positions_list (plus
    goal_position) with a margin, then report where goal_position falls
    within that box as an (fx, fy) fraction of its side length - used to
    pick a goal position on the reference (ground-truth) panel that the
    other panels can then be pinned to

    returns
    -------
    (xmin, xmax, ymin, ymax), (fx, fy)
    '''
    all_positions = np.concatenate([*positions_list, goal_position[None, :]])
    xmin_raw, ymin_raw = all_positions.min(axis=0)
    xmax_raw, ymax_raw = all_positions.max(axis=0)

    width = max(xmax_raw - xmin_raw, 1e-6)
    height = max(ymax_raw - ymin_raw, 1e-6)
    side = max(width, height) * (1 + margin)

    cx = (xmin_raw + xmax_raw) / 2
    cy = (ymin_raw + ymax_raw) / 2
    xmin, xmax = cx - side / 2, cx + side / 2
    ymin, ymax = cy - side / 2, cy + side / 2

    fx = np.clip((goal_position[0] - xmin) / side, 0.05, 0.95)
    fy = np.clip((goal_position[1] - ymin) / side, 0.05, 0.95)

    return (xmin, xmax, ymin, ymax), (fx, fy)


def _bounds_for_goal_fraction(positions_list, goal_position, fx: float, fy: float, margin: float = 0.3):
    '''
    fit a square box tightly around every point in positions_list (plus
    goal_position) that places goal_position at the (fx, fy) fraction of
    the box's side length, rather than fitting the box first and reading
    the fraction off afterwards (see _tight_bounds_and_goal_fraction) -
    used so every panel can zoom to its own data while keeping the goal at
    a consistent normalized position across panels
    '''
    all_positions = np.concatenate([*positions_list, goal_position[None, :]])
    dx = all_positions[:, 0] - goal_position[0]
    dy = all_positions[:, 1] - goal_position[1]

    width = max(np.max(np.clip(-dx, 0, None)) / fx, np.max(np.clip(dx, 0, None)) / (1 - fx), 1e-6)
    height = max(np.max(np.clip(-dy, 0, None)) / fy, np.max(np.clip(dy, 0, None)) / (1 - fy), 1e-6)
    side = max(width, height) * (1 + margin)

    xmin = goal_position[0] - fx * side
    xmax = goal_position[0] + (1 - fx) * side
    ymin = goal_position[1] - fy * side
    ymax = goal_position[1] + (1 - fy) * side

    return xmin, xmax, ymin, ymax


def plot_episode_trajectory_overlay(
    results_dir: str = RESULTS_DIR,
    episode: int = 1,
    save_path: str = PLOT_SAVE_PATH,
    window: int = 100,
    offset: int = 0,
):
    '''
    render a single static image of a window of a predator-prey episode,
    overlaying every step's dot positions in one frame: earlier positions
    are faded, later positions are fully saturated, so the image reads as
    the path each dot took over the window (mirrors the dot rendering in
    PredatorPreyEnv.render_rgb, but collapsed across time instead of per-frame)

    the window runs for `window` steps starting at `offset` steps into the
    episode (offset=0 is the episode's first step)

    inputs
    ------
    results_dir:str
        directory containing the per-episode <idx>.npz result files
    episode:int
        which episode file to load (loads '<episode>.npz')
    save_path:str
        where to save the resulting figure
    window:int
        number of steps to plot starting from the window's start
    offset:int
        number of steps into the episode before the window starts
    '''
    predator_positions, _target_positions, target_positions, goal_position = load_episode_positions(
        results_dir=results_dir, episode=episode,
    )

    start_idx, end_idx = offset, offset + window
    _warn_if_window_clamped(start_idx, end_idx, target_positions.shape[0])

    predator_positions = {name: pos[start_idx:end_idx] for name, pos in predator_positions.items()}
    target_positions = target_positions[start_idx:end_idx]

    n_steps = target_positions.shape[0]
    fade = np.linspace(0.15, 1.0, n_steps)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('white')

    legend_handles = _draw_ground_truth(ax, predator_positions, target_positions, goal_position, fade)

    _set_zoom_bounds(ax, [*predator_positions.values(), target_positions], goal_position)
    ax.set_title(f'Episode Ground Truth (steps {start_idx}-{start_idx + n_steps - 1})', fontweight='bold')
    ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9, prop={'weight': 'bold', 'size': 12})

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    print(f'saved plot to {save_path}')


def plot_world_state_overlay(
    results_dir: str = RESULTS_DIR,
    episode: int = 1,
    save_path: str = WORLD_STATE_SAVE_PATH,
    window: int = 100,
    offset: int = 0,
):
    '''
    render one figure with 4 panels: a large ground-truth overlay on the
    left, and 3 stacked panels on the right - one per predator agent -
    each showing that agent's estimated world state: the agent's own
    position and the target/goal are ground truth (the agent observes
    these exactly), but its teammates' positions are drawn from that
    agent's belief (infos['__common__']['sampled_predictions']) instead of
    ground truth, so each right-hand panel is what that agent "thinks" the
    world looks like

    inputs
    ------
    results_dir:str
        directory containing the per-episode <idx>.npz result files
    episode:int
        which episode file to load (loads '<episode>.npz')
    save_path:str
        where to save the resulting figure
    window:int
        number of steps to plot starting from the window's start
    offset:int
        number of steps into the episode before the window starts
    '''
    full_predator_positions, target_positions_by_agent, target_positions, goal_position = load_episode_positions(
        results_dir=results_dir, episode=episode,
    )
    full_beliefs = load_episode_beliefs(results_dir=results_dir, episode=episode)

    predator_names = sorted(full_predator_positions)

    start_idx, end_idx = offset, offset + window
    _warn_if_window_clamped(start_idx, end_idx, target_positions.shape[0])

    predator_positions = {name: pos[start_idx:end_idx] for name, pos in full_predator_positions.items()}
    target_positions_by_agent = {
        name: pos[start_idx:end_idx] for name, pos in target_positions_by_agent.items()
    }
    target_positions = target_positions[start_idx:end_idx]
    beliefs = {
        observer: {target: pos[start_idx:end_idx] for target, pos in targets.items()}
        for observer, targets in full_beliefs.items()
    }

    n_steps = target_positions.shape[0]
    fade = np.linspace(0.15, 1.0, n_steps)

    # each observer's estimated world state: its own position is ground
    # truth, teammates come from its belief
    estimated_positions_by_observer = {
        observer: {
            name: predator_positions[name] if name == observer else beliefs[observer][name]
            for name in predator_names
        }
        for observer in predator_names
    }

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(len(predator_names), 3, width_ratios=[1.3, 1, 1])
    ax_left = fig.add_subplot(gs[:, 0])
    right_axes = [fig.add_subplot(gs[i, 1]) for i in range(len(predator_names))]
    error_axes = [fig.add_subplot(gs[i, 2]) for i in range(len(predator_names))]

    # fit tightly to the ground-truth panel's own data, then read off where
    # the goal naturally falls (fx, fy) so every other panel can zoom to
    # its own data while pinning the goal to that same normalized spot
    ground_truth_positions = [*predator_positions.values(), target_positions]
    (xmin, xmax, ymin, ymax), (fx, fy) = _tight_bounds_and_goal_fraction(
        ground_truth_positions, goal_position,
    )

    legend_handles = _draw_ground_truth(ax_left, predator_positions, target_positions, goal_position, fade)
    ax_left.set_xlim(xmin, xmax)
    ax_left.set_ylim(ymin, ymax)
    ax_left.set_aspect('equal')
    ax_left.set_title(
        f'Episode Ground Truth (steps {start_idx}-{start_idx + n_steps - 1})', fontweight='bold', fontsize=14,
    )
    ax_left.legend(handles=legend_handles, loc='upper right', framealpha=0.9, prop={'weight': 'bold', 'size': 12})

    full_step_axis = np.arange(full_predator_positions[predator_names[0]].shape[0])

    for ax, error_ax, observer in zip(right_axes, error_axes, predator_names):
        observer_target_positions = target_positions_by_agent[observer]
        observer_positions = [*estimated_positions_by_observer[observer].values(), observer_target_positions]
        _draw_ground_truth(
            ax, estimated_positions_by_observer[observer], observer_target_positions, goal_position, fade,
        )
        oxmin, oxmax, oymin, oymax = _bounds_for_goal_fraction(observer_positions, goal_position, fx, fy)
        ax.set_xlim(oxmin, oxmax)
        ax.set_ylim(oymin, oymax)
        ax.set_aspect('equal')
        ax.set_title(f'{observer.capitalize()} Belief', fontweight='bold', fontsize=11)

        # belief error per teammate over the FULL episode (not just the
        # window plotted on the left/middle panels), same color as that
        # teammate's dots
        for target_name in predator_names:
            if target_name == observer:
                continue
            color = PREDATOR_COLORS[predator_names.index(target_name) % len(PREDATOR_COLORS)]
            error = np.linalg.norm(
                full_predator_positions[target_name] - full_beliefs[observer][target_name], axis=1,
            )

            error_ax.plot(full_step_axis, error, color=color, linewidth=1.5, label=target_name)

        error_ax.axvspan(start_idx, end_idx - 1, color='grey', alpha=0.15, zorder=0)
        error_ax.set_ylim(bottom=0)
        error_ax.set_ylabel('belief error')
        error_ax.set_title(f'{observer.capitalize()} Error', fontweight='bold', fontsize=11)
        error_ax.legend(prop={'weight': 'bold', 'size': 8})

    error_axes[-1].set_xlabel('step')

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    print(f'saved plot to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=1,
                         help='which episode file to load (loads "<episode>.npz")')
    parser.add_argument('--window', type=int, default=100,
                         help='number of steps to plot starting from the window start')
    parser.add_argument('--offset', type=int, default=0,
                         help='steps into the episode before the window starts')
    parser.add_argument('--world-state', action='store_true',
                         help='render the 4-panel ground-truth + per-agent estimated-world-state figure')
    parser.add_argument('--world-state-save-path', type=str, default=WORLD_STATE_SAVE_PATH,
                         help='where to save the 4-panel world-state figure')
    args = parser.parse_args()

    plot_episode_trajectory_overlay(
        episode=args.episode,
        window=args.window,
        offset=args.offset,
    )

    if args.world_state:
        plot_world_state_overlay(
            episode=args.episode,
            save_path=args.world_state_save_path,
            window=args.window,
            offset=args.offset,
        )
