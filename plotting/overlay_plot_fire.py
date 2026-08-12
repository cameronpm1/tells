import argparse
import itertools
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path


RESULTS_DIR = '/home/cameron/tells/logs/marl/fire_fully_obs/checkpoint10000/results'
#RESULTS_DIR = '/home/cameron/tells/logs/marl/fire_fully_obs/checkpoint10000/results_pf'
RESULTS_DIR = '/home/cameron/tells/logs/marl/fire_fully_obs/controller_results'
PLOT_SAVE_PATH = '/home/cameron/tells/test_fire.png'
WORLD_STATE_SAVE_PATH = '/home/cameron/tells/test_fire_world_state.png'

# which <EPISODE>.npz test run in RESULTS_DIR to plot, and which window of
# steps within it - edit these directly, or override with --episode/
# --window/--offset on the command line
EPISODE = 1
WINDOW = 60
OFFSET = 0

# matches DroneFireSim.GRID_SIZE (envs/marl/fire_env.py) - kept as a local
# constant rather than importing the env module, since that pulls in
# gymnasium/ray just to plot a grid of numbers
GRID_SIZE = 60

# each drone gets its own shade of blue so overlapping trails stay legible
# (mirrors the other overlay_plot_*.py files); the fire grid states use a
# darker forest green for unburned cells and a slightly darker red for
# burning cells than DroneFireSim.COLORS' pure (0,255,0)/(255,0,0), so the
# render reads less like a raw debug image and more like a rendered scene
DRONE_COLORS = ['#1f3fae', '#2f6fc9', '#4a93cf', '#6cb5d9']
UNBURNED_COLOR = '#1e4d2e'
BURNING_COLOR = '#a4231f'
BURNED_COLOR = '#101010'
GRID_LINE_COLOR = 'black'

# DroneFireSim.fire_state only ever takes GREEN(0)/RED(1)/BLACK(2) - see
# DroneFireSim.GREEN/RED/BLACK/WHITE/BLUE and initialize_fire/
# extinguish_fire/fire_controller, none of which ever write WHITE or BLUE
# into fire_state (those only appear in the padded local observation window)
FIRE_CMAP = ListedColormap([UNBURNED_COLOR, BURNING_COLOR, BURNED_COLOR])
FIRE_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], FIRE_CMAP.N)

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'


def load_episode_positions(results_dir: str, episode: int = 1):
    '''
    load a single episode's eval results and reconstruct each drone's
    absolute (row, col) position and the ground-truth fire grid at every
    step

    drone positions come from infos['__common__']['decomposed_obs'][agent]
    ['self_pos'] (see DroneFireEnv.step, which stashes each agent's own
    decomposed obs - fire/team/self_pos - into info before the noise-free
    positions get folded into the packaged observation); fire_env never
    perturbs self_pos, so this is exact ground truth, not a noisy reading

    the fire grid comes from infos['target'], which DroneFireEnv._compute_obs
    sets to self.env.fire_state directly (the full (GRID_SIZE, GRID_SIZE)
    simulation grid) rather than any agent's cropped observation window

    inputs
    ------
    results_dir:str
        directory containing the per-episode <idx>.npz result files
    episode:int
        which episode file to load (loads '<episode>.npz')

    returns
    -------
    drone_positions:dict of agent -> (T, 2) array of (row, col) positions
    fire_grids:(T, GRID_SIZE, GRID_SIZE) array of the ground-truth fire grid
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)

    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)
    if not steps:
        raise ValueError(f'no steps found in {npz_path}')

    first_common = data[steps[0]][4]['__common__']
    agent_names = sorted(first_common['decomposed_obs'].keys())

    drone_positions = {name: [] for name in agent_names}
    fire_grids = []

    for step in steps:
        _obs, _rewards, _terminations, _truncations, infos = data[step]
        decomposed_obs = infos['__common__']['decomposed_obs']

        for name in agent_names:
            drone_positions[name].append(np.asarray(decomposed_obs[name]['self_pos']))

        fire_grids.append(np.asarray(infos['target']))

    drone_positions = {name: np.stack(pos) for name, pos in drone_positions.items()}
    fire_grids = np.stack(fire_grids)

    return drone_positions, fire_grids


def _best_teammate_permutation(estimate, true_relative):
    '''
    unlike the belief models in the drones/football/predator_prey envs,
    DroneFireEnv's belief model doesn't commit its output to a fixed
    teammate-slot order - DroneFireEnv.team_error (envs/marl/fire_env.py)
    resolves this itself by checking every permutation of the true
    teammate order against the model's (fixed-slot) prediction and keeping
    whichever permutation minimizes total distance; this does the same
    thing for a single (step, observer) pair so a given predicted slot can
    be labeled with a specific teammate name for plotting/error purposes

    inputs
    ------
    estimate:(N, 2) array, the belief model's raw (fixed-slot) prediction
    true_relative:(N, 2) array, ground-truth relative teammate positions,
        in `others` order (see load_episode_beliefs)

    returns
    -------
    tuple of length N: perm such that estimate[i] best matches
    true_relative[perm[i]] - i.e. estimate slot i should be labeled
    others[perm[i]]
    '''
    best_perm, best_dist = None, np.inf

    for perm in itertools.permutations(range(len(true_relative))):
        dist = np.linalg.norm(estimate - true_relative[list(perm)], axis=1).sum()
        if dist < best_dist:
            best_dist = dist
            best_perm = perm

    return best_perm


def load_episode_beliefs(results_dir: str, episode: int = 1):
    '''
    load a single episode's eval results and, for each step, convert every
    drone's belief about its teammates (infos['__common__']['sampled_predictions'],
    produced by the belief model) into absolute positions

    sampled_predictions[observer] is that observer's (fixed-slot) estimate
    of decomposed_obs[observer]['team'] - see _best_teammate_permutation
    for how a slot gets matched up with a specific teammate name, since
    DroneFireEnv's belief model doesn't guarantee a fixed slot ordering

    each estimate is anchored to the observer's own ground-truth position
    (decomposed_obs[observer]['self_pos'], never noised - see
    DroneFireEnv._compute_obs) to recover an absolute estimated position
    for each teammate

    inputs
    ------
    results_dir:str
        directory containing the per-episode <idx>.npz result files
    episode:int
        which episode file to load (loads '<episode>.npz')

    returns
    -------
    dict: observer -> {target_drone: (T, 2) array of estimated positions}
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)

    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)
    if not steps:
        raise ValueError(f'no steps found in {npz_path}')

    first_common = data[steps[0]][4]['__common__']
    agent_names = sorted(first_common['decomposed_obs'].keys())

    first_predictions = first_common.get('sampled_predictions', {})
    if not all(agent in first_predictions for agent in agent_names):
        raise ValueError(
            f"{npz_path} has no per-agent 'sampled_predictions' (found keys "
            f'{list(first_predictions.keys())} at step {steps[0]}) - this looks like a '
            f"particle-filter eval run, which only records an aggregate scalar 'belief_error' "
            f'per step rather than per-agent position estimates; plot_world_state_overlay needs '
            f'a results directory with populated sampled_predictions (e.g. a non-"_pf" results '
            f'directory) to plot per-agent belief panels/error curves'
        )

    estimated = {observer: {target: [] for target in agent_names if target != observer} for observer in agent_names}

    for step in steps:
        _obs, _rewards, _terminations, _truncations, infos = data[step]
        common = infos['__common__']
        decomposed_obs = common['decomposed_obs']
        sampled_predictions = common['sampled_predictions']

        for observer in agent_names:
            others = [agent for agent in agent_names if agent != observer]
            observer_pos = np.asarray(decomposed_obs[observer]['self_pos'], dtype=float)
            true_relative = np.asarray(decomposed_obs[observer]['team'], dtype=float)
            estimate = np.asarray(sampled_predictions[observer], dtype=float).reshape(len(others), -1)

            perm = _best_teammate_permutation(estimate, true_relative)

            for slot_idx, true_idx in enumerate(perm):
                target = others[true_idx]
                estimated[observer][target].append(observer_pos + estimate[slot_idx])

    return {
        observer: {target: np.stack(positions) for target, positions in targets.items()}
        for observer, targets in estimated.items()
    }


def _warn_if_window_clamped(start_idx: int, end_idx: int, total_steps: int):
    '''
    print a warning if the requested [start_idx, end_idx) window runs past
    the end of the episode - the plotted data (and title) end up clamped
    to the steps that actually exist rather than the requested range
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


def _draw_fire_grid(ax, grid, cell_size: float = 1.0):
    '''
    draw the fire grid as a raster (forest green/burning red/burned black
    per DroneFireSim cell state) plus fine black gridlines at every cell
    boundary, so the render reads as a proper grid of cells rather than the
    plain 61x61 pixel dump render_fire_with_drones produces

    grid is drawn in the same (row, col) data coordinates drone positions
    are recorded in, so drone markers plotted afterward on the same axes
    line up with the cells they're standing on without any extra transform
    '''
    extent = (-0.5, GRID_SIZE - 0.5, GRID_SIZE - 0.5, -0.5)
    ax.imshow(grid, cmap=FIRE_CMAP, norm=FIRE_NORM, interpolation='nearest', extent=extent, zorder=1)

    ax.set_xticks(np.arange(-0.5, GRID_SIZE, cell_size), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_SIZE, cell_size), minor=True)
    ax.grid(which='minor', color=GRID_LINE_COLOR, linewidth=0.32, alpha=0.5, zorder=2)
    ax.tick_params(which='minor', length=0)
    ax.set_xticks([])
    ax.set_yticks([])


def _rounded_x_path(center, half_width: float, arm_length: float, corner_radius: float):
    '''
    build a matplotlib Path for a single solid "x" glyph: four thick arms
    crossing at center, with the four concave corners where adjacent arms
    meet smoothed into a rounded fillet (quadratic Bezier tangent to both
    edges) instead of a sharp notch, so the shape reads as one puffy x
    rather than four separate bars

    the outline is built axis-aligned (arms along +-x/+-y, i.e. a "+") and
    then rotated 45 degrees so the arms point diagonally, matching an "x"
    '''
    half_width = min(half_width, arm_length * 0.9)
    corner_radius = min(corner_radius, half_width * 0.95, (arm_length - half_width) * 0.95)

    # 12 outline vertices of a "+" (unrotated), going counter-clockwise;
    # odd indices are the concave inner corners between arms
    raw_points = [
        (half_width, arm_length), (half_width, half_width), (arm_length, half_width),
        (arm_length, -half_width), (half_width, -half_width), (half_width, -arm_length),
        (-half_width, -arm_length), (-half_width, -half_width), (-arm_length, -half_width),
        (-arm_length, half_width), (-half_width, half_width), (-half_width, arm_length),
    ]
    concave_idxs = {1, 4, 7, 10}

    theta = np.radians(45)
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    points = [rotation @ np.array(p) for p in raw_points]

    n = len(points)
    verts = []
    codes = []

    for i, p in enumerate(points):
        if i in concave_idxs:
            dir_in = points[i - 1] - p
            dir_in /= np.linalg.norm(dir_in)
            dir_out = points[(i + 1) % n] - p
            dir_out /= np.linalg.norm(dir_out)

            entry = p + dir_in * corner_radius
            exit_ = p + dir_out * corner_radius

            codes.append(Path.MOVETO if not codes else Path.LINETO)
            verts.append(entry)
            codes.append(Path.CURVE3)
            verts.append(p)
            codes.append(Path.CURVE3)
            verts.append(exit_)
        else:
            codes.append(Path.MOVETO if not codes else Path.LINETO)
            verts.append(p)

    verts.append(verts[0])
    codes.append(Path.CLOSEPOLY)

    verts = np.array(verts) + np.asarray(center)
    return Path(verts, codes)


def _draw_drone_icon(ax, center, color, size: float = 1.1, alpha: float = 1.0, zorder: float = 4,
                      rotor_radius: float = 0.26, rotor_linewidth: float = 1.8):
    '''
    draw a small top-down drone icon at center: a compact "pillow" body
    with concave-rounded sides (a wide/short instance of the same smoothed
    "x" glyph as _rounded_x_path, so its 4 corners come to a point rather
    than being a plain circle), a thin arm from each of those corners out
    to an unfilled rotor ring - standing in for a drone instead of the
    plain blue "x" render_fire_with_drones/DroneFireEnv.render_rgb use

    rotor_radius/rotor_linewidth are fixed (not scaled by size) so shrinking
    size makes the body/arms smaller while the rotor rings stay put
    '''
    body_reach = size * 0.36
    body_half_width = size * 0.17
    body_corner_radius = size * 0.3
    arm_length = size * 0.55
    arm_linewidth = max(size * 0.9, 1.0)

    body_path = _rounded_x_path(center, body_half_width, body_reach, body_corner_radius)
    ax.add_patch(PathPatch(body_path, facecolor=color, edgecolor='none', alpha=alpha, zorder=zorder))

    cx, cy = center
    for angle_deg in (45, 135, 225, 315):
        angle = np.radians(angle_deg)
        dx, dy = np.cos(angle), np.sin(angle)
        start = (cx + dx * body_reach, cy + dy * body_reach)
        tip = (cx + dx * arm_length, cy + dy * arm_length)

        ax.plot([start[0], tip[0]], [start[1], tip[1]], color=color, linewidth=arm_linewidth,
                 alpha=alpha, zorder=zorder, solid_capstyle='round')
        ax.add_patch(Circle(tip, rotor_radius, facecolor='none', edgecolor=color,
                             linewidth=rotor_linewidth, alpha=alpha, zorder=zorder + 0.1))


def _draw_ground_truth(ax, drone_positions, fire_grids, fade, trail_icon_scale: float = 0.55,
                        icon_size: float = 1.1):
    '''
    draw the fire grid (at its final, i.e. most recent, step - like the
    goal boxes in overlay_plot_drones.py being drawn at their current
    position) and the faded -> opaque drone-icon trails for every drone
    onto ax; returns the legend handles

    each trail step is drawn as its own (smaller, per trail_icon_scale)
    drone icon rather than a plain dot, with alpha taken from `fade` - the
    oldest step is the most transparent and alpha rises toward the most
    recent step, which is drawn separately afterward as a full-size,
    fully opaque icon
    '''
    _draw_fire_grid(ax, fire_grids[-1])

    legend_handles = [
        Line2D([0], [0], marker='s', color='none', markerfacecolor=UNBURNED_COLOR,
               markeredgecolor='none', markersize=11, label='unburned'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor=BURNING_COLOR,
               markeredgecolor='none', markersize=11, label='burning'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor=BURNED_COLOR,
               markeredgecolor='none', markersize=11, label='burned'),
    ]

    for idx, (name, positions) in enumerate(drone_positions.items()):
        color = DRONE_COLORS[idx % len(DRONE_COLORS)]

        # positions are (row, col) - plotted as (x=col, y=row) to match
        # _draw_fire_grid's imshow extent
        rows, cols = positions[:, 0], positions[:, 1]

        for t in range(len(positions) - 1):
            # skip trail steps sitting on the same cell as the current
            # position - that icon is drawn full-size/opaque below and a
            # faded copy directly under it just muddies the color
            if rows[t] == rows[-1] and cols[t] == cols[-1]:
                continue
            _draw_drone_icon(ax, (cols[t], rows[t]), color, size=icon_size * trail_icon_scale,
                              alpha=fade[t], zorder=3)
        _draw_drone_icon(ax, (cols[-1], rows[-1]), color, size=icon_size, zorder=4)

        legend_handles.append(Line2D([0], [0], marker='P', color='none', markerfacecolor=color,
                                      markeredgecolor='none', markersize=11, label=name))

    return legend_handles


def plot_episode_trajectory_overlay(
    results_dir: str = RESULTS_DIR,
    episode: int = EPISODE,
    save_path: str = PLOT_SAVE_PATH,
    window: int = WINDOW,
    offset: int = OFFSET,
):
    '''
    render a single static high-resolution image of a window of a fire
    episode, overlaying every step's drone positions in one frame: earlier
    positions are faded dots, the most recent position is a small drone
    icon - drawn over the fire grid's most recent state (forest green
    unburned, red burning, black burned cells) with fine gridlines at every
    cell boundary

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
    drone_positions, fire_grids = load_episode_positions(results_dir=results_dir, episode=episode)

    start_idx, end_idx = offset, offset + window
    _warn_if_window_clamped(start_idx, end_idx, fire_grids.shape[0])

    drone_positions = {name: pos[start_idx:end_idx] for name, pos in drone_positions.items()}
    fire_grids = fire_grids[start_idx:end_idx]

    n_steps = fire_grids.shape[0]
    # the most recent step is always drawn separately as a full-size icon
    # (see _draw_ground_truth), so capping the top of this range just under
    # 1.0 keeps the second-most-recent trail dot barely more faded than the
    # current position rather than visually identical
    fade = np.linspace(0.15, 0.9, n_steps)

    fig, ax = plt.subplots(figsize=(11, 11), dpi=200)

    legend_handles = _draw_ground_truth(ax, drone_positions, fire_grids, fade)

    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(GRID_SIZE - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(f'Fire Episode Ground Truth (steps {start_idx}-{start_idx + n_steps - 1})', fontweight='bold')
    ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9, prop={'weight': 'bold', 'size': 11})

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f'saved plot to {save_path}')


def plot_world_state_overlay(
    results_dir: str = RESULTS_DIR,
    episode: int = EPISODE,
    save_path: str = WORLD_STATE_SAVE_PATH,
    window: int = WINDOW,
    offset: int = OFFSET,
):
    '''
    render one figure with a large ground-truth overlay on the left, and
    one stacked panel per drone on the right - each showing that drone's
    estimated world state: the drone's own position is ground truth (and
    the fire grid is ground truth on every panel - it's directly observed,
    not predicted), but its teammates' positions are drawn from that
    drone's belief (infos['__common__']['sampled_predictions'], via
    load_episode_beliefs) instead of ground truth, so each right-hand
    panel is what that drone "thinks" its teammates' positions are; a
    third column plots that drone's belief error per teammate over the
    full episode

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
    full_drone_positions, full_fire_grids = load_episode_positions(results_dir=results_dir, episode=episode)
    full_beliefs = load_episode_beliefs(results_dir=results_dir, episode=episode)

    drone_names = sorted(full_drone_positions)

    start_idx, end_idx = offset, offset + window
    _warn_if_window_clamped(start_idx, end_idx, full_fire_grids.shape[0])

    drone_positions = {name: pos[start_idx:end_idx] for name, pos in full_drone_positions.items()}
    fire_grids = full_fire_grids[start_idx:end_idx]
    beliefs = {
        observer: {target: pos[start_idx:end_idx] for target, pos in targets.items()}
        for observer, targets in full_beliefs.items()
    }

    n_steps = fire_grids.shape[0]
    fade = np.linspace(0.15, 0.9, n_steps)

    # each observer's estimated world state: its own position is ground
    # truth, teammates come from its belief
    estimated_positions_by_observer = {
        observer: {
            name: drone_positions[name] if name == observer else beliefs[observer][name]
            for name in drone_names
        }
        for observer in drone_names
    }

    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(len(drone_names), 3, width_ratios=[1.3, 1, 1])
    ax_left = fig.add_subplot(gs[:, 0])
    right_axes = [fig.add_subplot(gs[i, 1]) for i in range(len(drone_names))]
    error_axes = [fig.add_subplot(gs[i, 2]) for i in range(len(drone_names))]

    legend_handles = _draw_ground_truth(ax_left, drone_positions, fire_grids, fade)
    ax_left.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax_left.set_ylim(GRID_SIZE - 0.5, -0.5)
    ax_left.set_aspect('equal')
    ax_left.set_title(
        f'Fire Episode Ground Truth (steps {start_idx}-{start_idx + n_steps - 1})', fontweight='bold', fontsize=14,
    )
    ax_left.legend(handles=legend_handles, loc='upper right', framealpha=0.9, prop={'weight': 'bold', 'size': 11})

    full_step_axis = np.arange(full_fire_grids.shape[0])
    # the requested window can run past the episode's actual length (see
    # _warn_if_window_clamped) - cap the highlighted span at the last real
    # step so it doesn't stretch the x-axis out past where the error lines
    # (plotted over full_step_axis, i.e. already just the real steps) end
    highlight_end_idx = min(end_idx, full_fire_grids.shape[0]) - 1

    for ax, error_ax, observer in zip(right_axes, error_axes, drone_names):
        # belief panels reuse the same (ground-truth) fire grid - it's
        # never predicted, only teammate positions are
        _draw_ground_truth(ax, estimated_positions_by_observer[observer], fire_grids, fade, icon_size=0.6)
        ax.set_xlim(-0.5, GRID_SIZE - 0.5)
        ax.set_ylim(GRID_SIZE - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{observer.capitalize()} Belief', fontweight='bold', fontsize=11)

        # belief error per teammate over the FULL episode (not just the
        # window plotted on the left/middle panels), same color as that
        # teammate's icon
        for target_name in drone_names:
            if target_name == observer:
                continue
            color = DRONE_COLORS[drone_names.index(target_name) % len(DRONE_COLORS)]
            error = np.linalg.norm(
                full_drone_positions[target_name] - full_beliefs[observer][target_name], axis=1,
            )

            error_ax.plot(full_step_axis, error, color=color, linewidth=1.5, label=target_name)

        error_ax.axvspan(start_idx, highlight_end_idx, color='grey', alpha=0.15, zorder=0)
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
    parser.add_argument('--episode', type=int, default=EPISODE,
                         help='which episode file to load (loads "<episode>.npz")')
    parser.add_argument('--window', type=int, default=WINDOW,
                         help='number of steps to plot starting from the window start')
    parser.add_argument('--offset', type=int, default=OFFSET,
                         help='steps into the episode before the window starts')
    parser.add_argument('--world-state', action='store_true',
                         help='render the multi-panel ground-truth + per-drone estimated-world-state figure')
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
