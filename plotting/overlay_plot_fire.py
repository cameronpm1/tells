import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


RESULTS_DIR = '/home/cameron/tells/logs/marl/fire_fully_obs/checkpoint20000/results'
PLOT_SAVE_PATH = '/home/cameron/tells/test_fire.png'

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
    ax.grid(which='minor', color=GRID_LINE_COLOR, linewidth=0.25, alpha=0.35, zorder=2)
    ax.tick_params(which='minor', length=0)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_drone_icon(ax, center, color, size: float = 1.6, alpha: float = 1.0, zorder: float = 4):
    '''
    draw a small top-down quadcopter icon at center: a body dot with four
    thin arms extending diagonally out to four rotor dots, standing in for
    a drone instead of the plain blue "x" render_fire_with_drones/
    DroneFireEnv.render_rgb use
    '''
    cx, cy = center
    arm = size * 0.5
    rotor_radius = size * 0.24
    body_radius = size * 0.16
    arm_width = max(size * 0.7, 0.6)

    for dx, dy in [(arm, arm), (arm, -arm), (-arm, arm), (-arm, -arm)]:
        ax.plot([cx, cx + dx], [cy, cy + dy], color=color, linewidth=arm_width,
                 alpha=alpha, zorder=zorder, solid_capstyle='round')
        ax.add_patch(Circle((cx + dx, cy + dy), rotor_radius, facecolor=color,
                             edgecolor='none', alpha=alpha, zorder=zorder + 0.1))

    ax.add_patch(Circle((cx, cy), body_radius, facecolor=color, edgecolor='none',
                         alpha=alpha, zorder=zorder + 0.2))


def _draw_ground_truth(ax, drone_positions, fire_grids, fade, trail_size: float = 30, icon_size: float = 1.6):
    '''
    draw the fire grid (at its final, i.e. most recent, step - like the
    goal boxes in overlay_plot_drones.py being drawn at their current
    position) and the faded -> saturated trails for every drone onto ax;
    returns the legend handles
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
        cmap = _fade_cmap(color, f'drone_{idx}')

        # positions are (row, col) - plotted as (x=col, y=row) to match
        # _draw_fire_grid's imshow extent
        rows, cols = positions[:, 0], positions[:, 1]

        if len(positions) > 1:
            ax.scatter(cols[:-1], rows[:-1], color=cmap(fade[:-1]), s=trail_size, zorder=3)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=EPISODE,
                         help='which episode file to load (loads "<episode>.npz")')
    parser.add_argument('--window', type=int, default=WINDOW,
                         help='number of steps to plot starting from the window start')
    parser.add_argument('--offset', type=int, default=OFFSET,
                         help='steps into the episode before the window starts')
    args = parser.parse_args()

    plot_episode_trajectory_overlay(
        episode=args.episode,
        window=args.window,
        offset=args.offset,
    )
