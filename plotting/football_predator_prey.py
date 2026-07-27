import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle


#RESULTS_DIR = '/home/cameron/tells/logs/marl/football_fully_obs_slow/checkpoint5000/results'
RESULTS_DIR = '/home/cameron/tells/logs/marl/football_fully_obs_slow/checkpoint0_pretrain/results_pf'
PLOT_SAVE_PATH = '/home/cameron/tells/test_football.png'
WORLD_STATE_SAVE_PATH = '/home/cameron/tells/test_football_world_state.png'

# matches the entity colors drawn in CirclePass5v1Env.render_rgb (left team blue
# (40, 90, 255), right team/defender red (255, 60, 60)), with each passer given
# its own shade of blue so overlapping trails stay legible; the ball gets a dark
# neutral color so it stays visible against both the green pitch and the legend
AGENT_COLORS = ['#1f3fae', '#2f6fc9', '#4a93cf', '#6cb5d9', '#8ed0df']
TARGET_COLOR = '#ff3c3c'
BALL_COLOR = '#3a3a3a'

# pitch bounds/green match CirclePass5v1Env.render_rgb's x_min/x_max/y_min/y_max
# and background fill color (35, 120, 45)
PITCH_X_MIN, PITCH_X_MAX = -1.05, 1.05
PITCH_Y_MIN, PITCH_Y_MAX = -0.45, 0.45
PITCH_GREEN = '#23782d'
PITCH_LINE_COLOR = 'white'
PITCH_CENTER_CIRCLE_RADIUS = 0.09

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
    positions for every passer agent, the defender ("target", as seen by
    each agent), and the ball at each step

    positions are recovered from each agent's own observation, using
    obs_map['self_pos'] for the agent's own absolute position (never
    noised) and the relative slices obs_map['target_pos'] / obs_map['ball_pos']
    (both stored relative to that agent) to recover defender/ball position

    the defender and ball are reconstructed separately per agent (rather
    than once, shared) because noise/particle-filter belief pipelines give
    each agent a different observed/estimated reading - using one agent's
    reading for every panel would silently mislabel the others'

    true_target_positions comes from infos['target'] instead - the
    defender's own observation, as returned directly by the env (see
    RLLibWrapper.step / PFWrapper.step: `infos['target'] = obs['target']`
    before the 'target' key is stripped from the returned obs dict) - this
    is the actual defender position, not any agent's (possibly noisy or
    estimated) reading of it

    true_ball_positions comes from infos['__common__']['obs_no_noise']
    instead - a snapshot of every agent's own obs taken before per-step
    noise (team/target_pos/ball_pos) is applied (see RLLibWrapper.step /
    PFWrapper.step), so self_pos + ball_pos read off obs_no_noise recovers
    the ball's true absolute position rather than any agent's noisy reading

    inputs
    ------
    results_dir:str
        directory containing the per-episode <idx>.npz result files
    episode:int
        which episode file to load (loads '<episode>.npz')

    returns
    -------
    agent_positions:dict of agent -> (T, 2) array
    target_positions:dict of agent -> (T, 2) array, that agent's own
        observed defender position
    true_target_positions:(T, 2) array, the defender's own reported position
    ball_positions:dict of agent -> (T, 2) array, that agent's own
        observed ball position
    true_ball_positions:(T, 2) array, the ball's true position
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)

    obs_map = data['obs_map'].item()
    self_pos_slice = obs_map['self_pos']
    target_pos_slice = obs_map['target_pos']
    ball_pos_slice = obs_map['ball_pos']

    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)
    if not steps:
        raise ValueError(f'no steps found in {npz_path}')

    first_obs = data[steps[0]][0]
    agent_names = sorted(first_obs.keys())

    agent_positions = {name: [] for name in agent_names}
    target_positions = {name: [] for name in agent_names}
    ball_positions = {name: [] for name in agent_names}
    true_target_positions = []
    true_ball_positions = []

    for step in steps:
        obs, _rewards, _terminations, _truncations, infos = data[step]
        obs_no_noise = infos['__common__']['obs_no_noise']

        for name in agent_names:
            self_pos = np.asarray(obs[name][self_pos_slice])
            agent_positions[name].append(self_pos)
            target_positions[name].append(self_pos + np.asarray(obs[name][target_pos_slice]))
            ball_positions[name].append(self_pos + np.asarray(obs[name][ball_pos_slice]))

        true_target_positions.append(np.asarray(infos['target'][self_pos_slice]))

        ref_obs = np.asarray(obs_no_noise[agent_names[0]])
        true_ball_positions.append(ref_obs[self_pos_slice] + ref_obs[ball_pos_slice])

    agent_positions = {name: np.stack(pos) for name, pos in agent_positions.items()}
    target_positions = {name: np.stack(pos) for name, pos in target_positions.items()}
    ball_positions = {name: np.stack(pos) for name, pos in ball_positions.items()}
    true_target_positions = np.stack(true_target_positions)
    true_ball_positions = np.stack(true_ball_positions)

    return agent_positions, target_positions, true_target_positions, ball_positions, true_ball_positions


def load_episode_beliefs(results_dir: str, episode: int = 1):
    '''
    load a single episode's eval results and, for each step, convert every
    agent's belief about its teammates (infos['__common__']['sampled_predictions'],
    produced by the particle-filter/belief pipeline) into absolute positions

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
    there's no data left to clamp to, or if the requested window is empty
    (end_idx <= start_idx, e.g. --window 0), since that also leaves nothing
    to plot
    '''
    if end_idx <= start_idx:
        raise ValueError(
            f'requested window is empty (start step {start_idx}, end step {end_idx}) - '
            f'--window must be a positive number of steps'
        )

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


def _draw_pitch(ax, linewidth=2):
    '''
    draw a green pitch background with a boundary rectangle, center line,
    and center circle, matching CirclePass5v1Env.render_rgb's field markings
    (just in data coordinates/matplotlib patches instead of pixels), so the
    overlay reads as positions on the actual football pitch rather than a
    blank grid - drawn at zorder=1, below the dot trails (zorder>=3)

    linewidth lets the small per-observer belief panels use finer pitch
    lines than the main ground-truth panel
    '''
    ax.set_facecolor(PITCH_GREEN)

    ax.add_patch(Rectangle(
        (PITCH_X_MIN, PITCH_Y_MIN), PITCH_X_MAX - PITCH_X_MIN, PITCH_Y_MAX - PITCH_Y_MIN,
        fill=False, edgecolor=PITCH_LINE_COLOR, linewidth=linewidth, zorder=1,
    ))
    ax.plot([0, 0], [PITCH_Y_MIN, PITCH_Y_MAX], color=PITCH_LINE_COLOR, linewidth=linewidth, zorder=1)
    ax.add_patch(Circle(
        (0, 0), PITCH_CENTER_CIRCLE_RADIUS, fill=False, edgecolor=PITCH_LINE_COLOR, linewidth=linewidth, zorder=1,
    ))


def _draw_ground_truth(
    ax, agent_positions, target_positions, ball_positions, fade,
    trail_size=35, final_size=150, pitch_linewidth=2,
):
    '''
    draw the green pitch background, then the faded -> saturated dot trails
    for every passer agent, the defender ("target"), and the ball onto ax;
    returns the legend handles

    trail_size/final_size control the trail-dot and final-position-dot
    marker sizes respectively, and pitch_linewidth the pitch marking
    thickness, so the small per-observer belief panels can use smaller
    markers/finer lines than the main ground-truth panel
    '''
    _draw_pitch(ax, linewidth=pitch_linewidth)

    legend_handles = []

    for idx, (name, positions) in enumerate(agent_positions.items()):
        color = AGENT_COLORS[idx % len(AGENT_COLORS)]
        cmap = _fade_cmap(color, f'agent_{idx}')

        ax.scatter(positions[:, 0], positions[:, 1], color=cmap(fade), s=trail_size, zorder=3)
        ax.scatter(*positions[-1], color=color, s=final_size, zorder=4)

        legend_handles.append(Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
                                      markeredgecolor='none', markersize=9, label=name))

    target_cmap = _fade_cmap(TARGET_COLOR, 'target')
    ax.scatter(target_positions[:, 0], target_positions[:, 1], color=target_cmap(fade), s=trail_size, zorder=3)
    ax.scatter(*target_positions[-1], color=TARGET_COLOR, s=final_size, zorder=4)
    legend_handles.append(Line2D([0], [0], marker='o', color='none', markerfacecolor=TARGET_COLOR,
                                  markeredgecolor='none', markersize=9, label='target'))

    ball_cmap = _fade_cmap(BALL_COLOR, 'ball')
    ax.scatter(ball_positions[:, 0], ball_positions[:, 1], color=ball_cmap(fade), s=trail_size, zorder=3)
    ax.scatter(*ball_positions[-1], color=BALL_COLOR, s=final_size, zorder=5)
    legend_handles.append(Line2D([0], [0], marker='o', color='none', markerfacecolor=BALL_COLOR,
                                  markeredgecolor='none', markersize=9, label='ball'))

    return legend_handles


def _set_zoom_bounds(ax, positions_list, anchor_position):
    '''
    zoom ax in on anchor_position, sized to fit every point in
    positions_list plus a small margin
    '''
    all_positions = np.concatenate([*positions_list, anchor_position[None, :]])
    radius = np.max(np.linalg.norm(all_positions - anchor_position, axis=1))
    bound = radius * 1.05 + 0.05
    ax.set_xlim(anchor_position[0] - bound, anchor_position[0] + bound)
    ax.set_ylim(anchor_position[1] - bound, anchor_position[1] + bound)
    ax.set_aspect('equal')


def _tight_bounds_and_anchor_fraction(positions_list, anchor_position, margin: float = 0.3):
    '''
    fit a square box tightly around every point in positions_list (plus
    anchor_position) with a margin, then report where anchor_position falls
    within that box as an (fx, fy) fraction of its side length - used to
    pick an anchor position on the reference (ground-truth) panel that the
    other panels can then be pinned to

    returns
    -------
    (xmin, xmax, ymin, ymax), (fx, fy)
    '''
    all_positions = np.concatenate([*positions_list, anchor_position[None, :]])
    xmin_raw, ymin_raw = all_positions.min(axis=0)
    xmax_raw, ymax_raw = all_positions.max(axis=0)

    width = max(xmax_raw - xmin_raw, 1e-6)
    height = max(ymax_raw - ymin_raw, 1e-6)
    side = max(width, height) * (1 + margin)

    cx = (xmin_raw + xmax_raw) / 2
    cy = (ymin_raw + ymax_raw) / 2
    xmin, xmax = cx - side / 2, cx + side / 2
    ymin, ymax = cy - side / 2, cy + side / 2

    fx = np.clip((anchor_position[0] - xmin) / side, 0.05, 0.95)
    fy = np.clip((anchor_position[1] - ymin) / side, 0.05, 0.95)

    return (xmin, xmax, ymin, ymax), (fx, fy)


def plot_episode_trajectory_overlay(
    results_dir: str = RESULTS_DIR,
    episode: int = 1,
    save_path: str = PLOT_SAVE_PATH,
    window: int = 100,
    offset: int = 0,
):
    '''
    render a single static image of a window of a football rondo episode,
    overlaying every step's dot positions in one frame: earlier positions
    are faded, later positions are fully saturated, so the image reads as
    the path each dot took over the window (mirrors the dot rendering in
    CirclePass5v1Env.render_rgb, but collapsed across time instead of
    per-frame)

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
    agent_positions, _target_positions, target_positions, _ball_positions, ball_positions = load_episode_positions(
        results_dir=results_dir, episode=episode,
    )

    start_idx, end_idx = offset, offset + window
    _warn_if_window_clamped(start_idx, end_idx, target_positions.shape[0])

    agent_positions = {name: pos[start_idx:end_idx] for name, pos in agent_positions.items()}
    target_positions = target_positions[start_idx:end_idx]
    ball_positions = ball_positions[start_idx:end_idx]

    n_steps = target_positions.shape[0]
    fade = np.linspace(0.15, 1.0, n_steps)

    fig, ax = plt.subplots(figsize=(8, 8))

    legend_handles = _draw_ground_truth(ax, agent_positions, target_positions, ball_positions, fade)

    _set_zoom_bounds(ax, [*agent_positions.values(), target_positions, ball_positions], ball_positions[-1])
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
    render one figure with panels: a large ground-truth overlay on the
    left, and one stacked panel per passer agent on the right - each
    showing that agent's estimated world state: the agent's own position
    and the defender/ball are ground truth on the left panel (the agent
    observes the defender/ball only through its own, possibly noisy,
    reading), but its teammates' positions on the right panels are drawn
    from that agent's belief (infos['__common__']['sampled_predictions'])
    instead of ground truth, so each right-hand panel is what that agent
    "thinks" the world looks like

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
    (
        full_agent_positions, target_positions_by_agent, target_positions,
        ball_positions_by_agent, ball_positions,
    ) = load_episode_positions(results_dir=results_dir, episode=episode)
    full_beliefs = load_episode_beliefs(results_dir=results_dir, episode=episode)

    agent_names = sorted(full_agent_positions)

    start_idx, end_idx = offset, offset + window
    _warn_if_window_clamped(start_idx, end_idx, target_positions.shape[0])

    agent_positions = {name: pos[start_idx:end_idx] for name, pos in full_agent_positions.items()}
    target_positions_by_agent = {
        name: pos[start_idx:end_idx] for name, pos in target_positions_by_agent.items()
    }
    ball_positions_by_agent = {
        name: pos[start_idx:end_idx] for name, pos in ball_positions_by_agent.items()
    }
    target_positions = target_positions[start_idx:end_idx]
    ball_positions = ball_positions[start_idx:end_idx]
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
            name: agent_positions[name] if name == observer else beliefs[observer][name]
            for name in agent_names
        }
        for observer in agent_names
    }

    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(len(agent_names), 3, width_ratios=[1.3, 1.5, 1])
    ax_left = fig.add_subplot(gs[:, 0])
    right_axes = [fig.add_subplot(gs[i, 1]) for i in range(len(agent_names))]
    error_axes = [fig.add_subplot(gs[i, 2]) for i in range(len(agent_names))]

    # fit tightly to the ground-truth panel's own data, pinned to the ball's
    # final position (this panel isn't centered on the pitch, since the
    # rondo action can happen anywhere on it)
    ground_truth_positions = [*agent_positions.values(), target_positions, ball_positions]
    (xmin, xmax, ymin, ymax), _ = _tight_bounds_and_anchor_fraction(
        ground_truth_positions, ball_positions[-1],
    )

    legend_handles = _draw_ground_truth(ax_left, agent_positions, target_positions, ball_positions, fade)
    ax_left.set_xlim(xmin, xmax)
    ax_left.set_ylim(ymin, ymax)
    ax_left.set_aspect('equal')
    ax_left.set_title(
        f'Episode Ground Truth (steps {start_idx}-{start_idx + n_steps - 1})', fontweight='bold', fontsize=14,
    )
    ax_left.legend(handles=legend_handles, loc='upper right', framealpha=0.9, prop={'weight': 'bold', 'size': 12})

    full_step_axis = np.arange(full_agent_positions[agent_names[0]].shape[0])

    pitch_center = np.zeros(2)

    for ax, error_ax, observer in zip(right_axes, error_axes, agent_names):
        observer_target_positions = target_positions_by_agent[observer]
        observer_ball_positions = ball_positions_by_agent[observer]
        observer_positions = [
            *estimated_positions_by_observer[observer].values(),
            observer_target_positions, observer_ball_positions,
        ]
        _draw_ground_truth(
            ax, estimated_positions_by_observer[observer], observer_target_positions,
            observer_ball_positions, fade, trail_size=1, final_size=6, pitch_linewidth=0.5,
        )
        # center belief panels on the pitch's center (not the ball) so the
        # rondo action - which happens around the center circle - stays in
        # the middle of these small subplots
        _set_zoom_bounds(ax, observer_positions, pitch_center)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{observer.capitalize()} Belief', fontweight='bold', fontsize=11)

        # belief error per teammate over the FULL episode (not just the
        # window plotted on the left/middle panels), same color as that
        # teammate's dots
        for target_name in agent_names:
            if target_name == observer:
                continue
            color = AGENT_COLORS[agent_names.index(target_name) % len(AGENT_COLORS)]
            error = np.linalg.norm(
                full_agent_positions[target_name] - full_beliefs[observer][target_name], axis=1,
            )

            error_ax.plot(full_step_axis, error, color=color, linewidth=1.5, label=target_name)

        error_ax.axvspan(start_idx, end_idx - 1, color='grey', alpha=0.15, zorder=0)
        error_ax.set_ylim(bottom=0)
        error_ax.set_ylabel('belief error')
        error_ax.set_title(f'{observer.capitalize()} Error', fontweight='bold', fontsize=11)
        error_ax.legend(prop={'weight': 'bold', 'size': 8})

    error_axes[-1].set_xlabel('step')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.35)
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
                         help='render the multi-panel ground-truth + per-agent estimated-world-state figure')
    parser.add_argument('--world-state-save-path', type=str, default=WORLD_STATE_SAVE_PATH,
                         help='where to save the multi-panel world-state figure')
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
