import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


RESULTS_DIR = '/home/cameron/tells/logs/marl/drones_fully_obs/checkpoint1000/results'
#RESULTS_DIR = '/home/cameron/tells/logs/marl/drones_fully_obs/checkpoint1000/results_pf'
RESULTS_DIR = '/home/cameron/tells/logs/marl/drones_fully_obs/controller_results'
PLOT_SAVE_PATH = '/home/cameron/tells/test_drones.png'
WORLD_STATE_SAVE_PATH = '/home/cameron/tells/test_drones_world_state.png'

# which <EPISODE>.npz test run in RESULTS_DIR to plot, and which window of
# steps within it - edit these directly, or override with --episode/
# --window/--offset on the command line
EPISODE = 1
WINDOW = 60
OFFSET = 0

# same color scheme as overlay_plot_predator_prey.py: each protector drone
# gets its own shade so overlapping trails stay legible, the adversary
# ("target") is red - the goal boxes reuse the translucent green
# CaravanAviary._addObstacles renders them with (rgbaColor=[0.2, 0.85, 0.2, 0.35])
DRONE_COLORS = ['#4646aa', '#5daacf', '#8a5acf']
TARGET_COLOR = '#aa4646'
BOX_COLOR = '#32aa32'
BOX_ALPHA = 0.35

# matches CaravanAviary's default goal_box_half_extents (confs/drones/*.yaml)
GOAL_BOX_HALF_EXTENTS = (0.5, 0.5, 0.1)

# fixed view window, centered on the middle goal box every time (see
# _set_axes_bounds) - independent of episode/window, so the crop/zoom is
# identical run to run instead of stretching to fit wherever the drones flew.
# the 5 default goal boxes span +-4 in x (goal_box_spacing=1.0, half_extents
# 0.5, see CaravanAviary._initialize_box_state / confs/drones/*.yaml), so
# +-6 comfortably covers the whole line plus nearby drone activity
PLOT_HALF_EXTENT_XY = 6.0
PLOT_Z_MIN = 0.0
PLOT_Z_MAX = 2.5

# each drone is drawn as two thin crossing bars (an "X") extruded a shallow
# amount in z, rather than the flat dot markers the 2d envs use - trail
# markers interpolate size (see _drone_trail_arm_length) from
# DRONE_TRAIL_ARM_LENGTH at the oldest/most-faded step up to
# DRONE_FINAL_ARM_LENGTH once a step is fully opaque, so an opaque trail
# point is never smaller than the most recent position's marker
DRONE_TRAIL_ARM_LENGTH = 0.32
DRONE_FINAL_ARM_LENGTH = 0.7
DRONE_ARM_WIDTH = 0.13
DRONE_HEIGHT = 0.06

# camera: a shallow (low) elevation reads more like a ground-level pybullet
# view than matplotlib's default elev=30 top-down-ish angle, and zoom > 1
# fills more of the figure with the scene (see Axes3D.set_box_aspect) - since
# the view window is now fixed (PLOT_HALF_EXTENT_XY etc.) rather than fit to
# per-episode data, its aspect ratio is a fixed known quantity, so this only
# needed tuning once: verified (via per-box pixel-area checks looking for
# mplot3d's near-plane clipping degenerating a box into a sliver) safe up to
# ~zoom 2.2 for this aspect ratio, so this stays comfortably under that
CAMERA_ELEV = 25
CAMERA_AZIM = -50
CAMERA_ZOOM = 1.8

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
    load a single episode's eval results and reconstruct absolute (x, y, z)
    positions for every protector drone, the adversary ("target"), and the
    goal boxes at each step

    protector positions are recovered from each drone's own observation via
    obs_map['self_pos'] (never noised, see RLLibWrapper/PFWrapper.step,
    which only add noise to obs_map['target_pos'] and obs_map['team'])

    target_positions is that drone's own (possibly noisy - see
    RLLibWrapper.step/PFWrapper.step, which add noise to
    obs_map['target_pos']) reading of the adversary, recovered from
    self_pos + obs_map['target_pos']

    true_target_positions comes from infos['target'] instead - the
    adversary's own observation, as returned directly by the env (see
    RLLibWrapper.step / PFWrapper.step: `infos['target'] = obs['target']`
    before the 'target' key is stripped from the returned obs dict) - this
    is the actual adversary position, not any protector's (possibly noisy)
    reading of it

    box_positions is read off obs_map['target_goal'] (also never noised)
    via a single reference drone's observation - CaravanAviary.box_state
    defaults to zero velocity so these are normally static, but are
    reconstructed per step in case nonzero box velocities are ever used

    inputs
    ------
    results_dir:str
        directory containing the per-episode <idx>.npz result files
    episode:int
        which episode file to load (loads '<episode>.npz')

    returns
    -------
    drone_positions:dict of agent -> (T, 3) array
    target_positions:dict of agent -> (T, 3) array, that agent's own
        observed adversary position
    true_target_positions:(T, 3) array, the adversary's own reported position
    box_positions:(T, num_goal_boxes, 3) array
    '''
    npz_path = os.path.join(results_dir, f'{episode}.npz')
    data = np.load(npz_path, allow_pickle=True)

    obs_map = data['obs_map'].item()
    self_pos_slice = obs_map['self_pos']
    target_pos_slice = obs_map['target_pos']
    target_goal_slice = obs_map['target_goal']
    num_boxes = (target_goal_slice.stop - target_goal_slice.start) // 3

    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)
    if not steps:
        raise ValueError(f'no steps found in {npz_path}')

    first_obs = data[steps[0]][0]
    drone_names = sorted(first_obs.keys())

    drone_positions = {name: [] for name in drone_names}
    target_positions = {name: [] for name in drone_names}
    true_target_positions = []
    box_positions = []

    for step in steps:
        obs, _rewards, _terminations, _truncations, infos = data[step]

        for name in drone_names:
            self_pos = np.asarray(obs[name][self_pos_slice])
            drone_positions[name].append(self_pos)
            target_positions[name].append(self_pos + np.asarray(obs[name][target_pos_slice]))

        true_target_positions.append(np.asarray(infos['target'][self_pos_slice]))

        ref_obs = obs[drone_names[0]]
        ref_pos = np.asarray(ref_obs[self_pos_slice])
        box_positions.append(ref_pos + np.asarray(ref_obs[target_goal_slice]).reshape(num_boxes, 3))

    drone_positions = {name: np.stack(pos) for name, pos in drone_positions.items()}
    target_positions = {name: np.stack(pos) for name, pos in target_positions.items()}
    true_target_positions = np.stack(true_target_positions)
    box_positions = np.stack(box_positions)

    return drone_positions, target_positions, true_target_positions, box_positions


def load_episode_beliefs(results_dir: str, episode: int = 1):
    '''
    load a single episode's eval results and, for each step, convert every
    drone's belief about its teammates (infos['__common__']['sampled_predictions'],
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
    dict: observer -> {target_drone: (T, 3) array of estimated positions}
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
    drone_names = list(first_common['obs_no_noise'].keys())
    dim = (team_slice.stop - team_slice.start) // (len(drone_names) - 1)

    estimated = {observer: {target: [] for target in drone_names if target != observer} for observer in drone_names}

    for step in steps:
        _obs, _rewards, _terminations, _truncations, infos = data[step]
        common = infos['__common__']
        obs_no_noise = common['obs_no_noise']
        sampled_predictions = common['sampled_predictions']

        for observer in drone_names:
            others = [drone for drone in drone_names if drone != observer]
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


def _box_faces(center, half_extents, yaw_deg: float = 0.0):
    '''
    build the 6 quad faces of a cuboid centered at `center` with the given
    half-extents, optionally rotated yaw_deg about z - used both for the
    goal boxes and for the two crossing bars that make up a drone's "X"
    '''
    hx, hy, hz = half_extents
    signs = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=float)
    corners = signs * np.array([hx, hy, hz])

    if yaw_deg:
        theta = np.radians(yaw_deg)
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        corners = corners @ rotation.T

    corners = corners + np.asarray(center)

    face_idxs = [
        [0, 1, 2, 3], [4, 5, 6, 7],
        [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
    ]
    return [corners[idx] for idx in face_idxs]


def _draw_box(ax, center, half_extents, color, alpha: float, yaw_deg: float = 0.0, zorder: int = 1):
    poly = Poly3DCollection(
        _box_faces(center, half_extents, yaw_deg),
        facecolor=color, alpha=alpha, edgecolor='none', zorder=zorder,
    )
    ax.add_collection3d(poly)


def _draw_drone_marker(ax, center, color, alpha: float = 1.0, arm_length: float = DRONE_FINAL_ARM_LENGTH,
                        zorder: int = 3):
    '''
    draw a drone as a shallow-height 3d "X": two thin crossing bars,
    rotated +-45 degrees from center, standing in for the quadrotor's
    crossed arms but flattened to a shallow height so it reads clearly
    from any camera angle (mirrors goal_box_half_extents' shallow z extent)
    '''
    half_extents = (arm_length / 2, DRONE_ARM_WIDTH / 2, DRONE_HEIGHT / 2)
    _draw_box(ax, center, half_extents, color, alpha, yaw_deg=45, zorder=zorder)
    _draw_box(ax, center, half_extents, color, alpha, yaw_deg=-45, zorder=zorder)


def _drone_trail_arm_length(fade_value: float) -> float:
    '''
    interpolate a trail marker's arm length from DRONE_TRAIL_ARM_LENGTH (at
    fade_value=0, the oldest/most-faded step) up to DRONE_FINAL_ARM_LENGTH
    (at fade_value=1) - matching the color fade curve so an opaque trail
    point (fade_value near 1) renders at the same size as the most recent
    position's marker instead of staying pinned to the small trail size
    '''
    return DRONE_TRAIL_ARM_LENGTH + (DRONE_FINAL_ARM_LENGTH - DRONE_TRAIL_ARM_LENGTH) * fade_value


def _draw_ground_truth(ax, drone_positions, target_positions, box_positions, fade):
    '''
    draw the translucent green goal boxes (at their current, i.e. final-step,
    position - like the static goal marker in overlay_plot_predator_prey.py)
    and the faded -> saturated "X" trails for every protector drone and the
    adversary target onto ax; returns the legend handles
    '''
    legend_handles = []

    for box_center in box_positions[-1]:
        ground_center = np.array([box_center[0], box_center[1], GOAL_BOX_HALF_EXTENTS[2]])
        # drawn above the faded trail dots (zorder 3) so a box never reads as
        # partially "missing" where a trail happens to cross over it, but
        # below the solid final-position markers (zorder 4/5)
        _draw_box(ax, ground_center, GOAL_BOX_HALF_EXTENTS, BOX_COLOR, BOX_ALPHA, zorder=3.5)
    legend_handles.append(Line2D([0], [0], marker='s', color='none', markerfacecolor=BOX_COLOR,
                                  markeredgecolor='none', alpha=0.6, markersize=9, label='goal box'))

    for idx, (name, positions) in enumerate(drone_positions.items()):
        color = DRONE_COLORS[idx % len(DRONE_COLORS)]
        cmap = _fade_cmap(color, f'drone_{idx}')

        for t in range(len(positions) - 1):
            _draw_drone_marker(ax, positions[t], cmap(fade[t]), arm_length=_drone_trail_arm_length(fade[t]))
        _draw_drone_marker(ax, positions[-1], color, arm_length=DRONE_FINAL_ARM_LENGTH, zorder=4)

        legend_handles.append(Line2D([0], [0], marker='X', color='none', markerfacecolor=color,
                                      markeredgecolor='none', markersize=9, label=name))

    target_cmap = _fade_cmap(TARGET_COLOR, 'target')
    for t in range(len(target_positions) - 1):
        _draw_drone_marker(ax, target_positions[t], target_cmap(fade[t]), arm_length=_drone_trail_arm_length(fade[t]))
    _draw_drone_marker(ax, target_positions[-1], TARGET_COLOR, arm_length=DRONE_FINAL_ARM_LENGTH, zorder=5)
    legend_handles.append(Line2D([0], [0], marker='X', color='none', markerfacecolor=TARGET_COLOR,
                                  markeredgecolor='none', markersize=9, label='target'))

    return legend_handles


def _set_axes_bounds(ax, center_xy, zoom: float = CAMERA_ZOOM):
    '''
    fix the view to a constant-size window centered on the goal-box line
    (center_xy - see plot_episode_trajectory_overlay, which passes the
    middle goal box's position) rather than fitting to wherever the drones
    happen to fly - drone excursions vary a lot step to step, and scaling
    the crop to them made the zoom/centering inconsistent run to run (and,
    since the resulting box_aspect ratio varied with the data, occasionally
    tripped mplot3d's near-plane clipping for particular episodes); a fixed
    window is the same every time by construction, and its aspect ratio is
    a fixed known quantity so CAMERA_ZOOM only ever needs to be tuned once
    '''
    xmin, xmax = center_xy[0] - PLOT_HALF_EXTENT_XY, center_xy[0] + PLOT_HALF_EXTENT_XY
    ymin, ymax = center_xy[1] - PLOT_HALF_EXTENT_XY, center_xy[1] + PLOT_HALF_EXTENT_XY

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(PLOT_Z_MIN, PLOT_Z_MAX)
    ax.set_box_aspect((xmax - xmin, ymax - ymin, PLOT_Z_MAX - PLOT_Z_MIN), zoom=zoom)


def _strip_grid_and_ticks(ax):
    '''
    hide the 3d axes' gridlines, tick marks/labels, pane edges, and axis
    corner lines so the scene reads as a clean render rather than a
    data-plot grid
    '''
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis.line.set_color((0, 0, 0, 0))


def _draw_ground_truth_2d(ax, drone_positions, target_positions, box_positions, fade,
                           trail_size: float = 35, final_size: float = 150):
    '''
    2d birds-eye (top-down, x-y) counterpart to _draw_ground_truth - draws
    the goal boxes as flat green squares (their x/y footprint) and the
    faded -> saturated dot trails for every drone and the target, matching
    the style of overlay_plot_predator_prey.py's/overlay_plot_football.py's
    ground-truth panels; used for the small per-drone belief panels in
    plot_world_state_overlay, where the full 3d "X" rendering would be
    unreadable at that size

    trail_size/final_size control the trail-dot and final-position-dot
    marker sizes, so the small per-observer belief panels can use smaller
    markers than a full-size standalone 2d plot would

    returns the legend handles
    '''
    legend_handles = []

    hx, hy, _hz = GOAL_BOX_HALF_EXTENTS
    for box_center in box_positions[-1]:
        ax.add_patch(Rectangle(
            (box_center[0] - hx, box_center[1] - hy), 2 * hx, 2 * hy,
            facecolor=BOX_COLOR, alpha=BOX_ALPHA, edgecolor='none', zorder=1,
        ))
    legend_handles.append(Line2D([0], [0], marker='s', color='none', markerfacecolor=BOX_COLOR,
                                  markeredgecolor='none', alpha=0.6, markersize=9, label='goal box'))

    for idx, (name, positions) in enumerate(drone_positions.items()):
        color = DRONE_COLORS[idx % len(DRONE_COLORS)]
        cmap = _fade_cmap(color, f'drone_2d_{idx}')

        ax.scatter(positions[:, 0], positions[:, 1], color=cmap(fade), s=trail_size, zorder=3)
        ax.scatter(*positions[-1, :2], color=color, s=final_size, zorder=4)

        legend_handles.append(Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
                                      markeredgecolor='none', markersize=9, label=name))

    target_cmap = _fade_cmap(TARGET_COLOR, 'target_2d')
    ax.scatter(target_positions[:, 0], target_positions[:, 1], color=target_cmap(fade), s=trail_size, zorder=3)
    ax.scatter(*target_positions[-1, :2], color=TARGET_COLOR, s=final_size, zorder=5)
    legend_handles.append(Line2D([0], [0], marker='o', color='none', markerfacecolor=TARGET_COLOR,
                                  markeredgecolor='none', markersize=9, label='target'))

    return legend_handles


def _set_axes_bounds_2d(ax, center_xy, half_extent: float = PLOT_HALF_EXTENT_XY):
    '''
    2d counterpart to _set_axes_bounds - same fixed, centered-on-the-middle-
    box window, just without the z axis
    '''
    ax.set_xlim(center_xy[0] - half_extent, center_xy[0] + half_extent)
    ax.set_ylim(center_xy[1] - half_extent, center_xy[1] + half_extent)
    ax.set_aspect('equal')


def plot_episode_trajectory_overlay(
    results_dir: str = RESULTS_DIR,
    episode: int = EPISODE,
    save_path: str = PLOT_SAVE_PATH,
    window: int = WINDOW,
    offset: int = OFFSET,
):
    '''
    render a single static 3d image of a window of a drones episode,
    overlaying every step's "X"-marker positions in one frame: earlier
    positions are faded, later positions are fully saturated, so the image
    reads as the path each drone took over the window - drawn similarly to
    how pybullet renders the scene (translucent green goal boxes on the
    ground, drones as small crossed-arm markers)

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
    drone_positions, _target_positions, target_positions, box_positions = load_episode_positions(
        results_dir=results_dir, episode=episode,
    )

    start_idx, end_idx = offset, offset + window
    _warn_if_window_clamped(start_idx, end_idx, target_positions.shape[0])

    drone_positions = {name: pos[start_idx:end_idx] for name, pos in drone_positions.items()}
    target_positions = target_positions[start_idx:end_idx]
    box_positions = box_positions[start_idx:end_idx]

    # the most recent step is always drawn separately at full color/size (see
    # _draw_ground_truth) regardless of this array, so capping the top of this
    # range just under 1.0 keeps the second-most-recent trail step barely
    # more faded than the current position rather than visually identical
    n_steps = target_positions.shape[0]
    fade = np.linspace(0.1, 0.9, n_steps)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(projection='3d')
    ax.set_facecolor('white')
    # Axes3D's default automatic depth-sorting (computed_zorder) mis-sorts
    # when the large flat goal boxes intersect the drone trail markers,
    # rendering parts of a box as a missing wedge; disabling it makes layering
    # follow each artist's explicit zorder instead (boxes below, drones above)
    ax.computed_zorder = False
    ax.view_init(elev=CAMERA_ELEV, azim=CAMERA_AZIM)

    legend_handles = _draw_ground_truth(ax, drone_positions, target_positions, box_positions, fade)

    center_box = box_positions[-1][box_positions.shape[1] // 2]
    _set_axes_bounds(ax, center_box)
    _strip_grid_and_ticks(ax)
    ax.set_title(f'Episode Ground Truth (steps {start_idx}-{start_idx + n_steps - 1})', fontweight='bold')
    ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9, prop={'weight': 'bold', 'size': 12})

    # the 3d axes' own subplot margins leave a lot of unused figure space
    # around whatever the box_aspect/zoom renders, so push it out manually
    # instead of the usual fig.tight_layout()
    ax.set_position((0.02, 0.02, 0.96, 0.9))
    fig.savefig(save_path)
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
    render one figure with the 3d ground-truth overlay on the left, and one
    stacked 2d birds-eye panel per protector drone in the middle - each
    showing that drone's estimated world state: the drone's own position is
    ground truth and the target/boxes come from that drone's own (possibly
    noisy) reading, but its teammates' positions are drawn from that
    drone's belief (infos['__common__']['sampled_predictions']) instead of
    ground truth, so each middle panel is what that drone "thinks" the
    world looks like; the right column plots that drone's belief error per
    teammate over the full episode (mirrors
    overlay_plot_predator_prey.py/overlay_plot_football.py's world-state
    overlay, with the ground-truth panel kept 3d instead of 2d)

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
    full_drone_positions, full_target_positions_by_drone, target_positions, box_positions = load_episode_positions(
        results_dir=results_dir, episode=episode,
    )
    full_beliefs = load_episode_beliefs(results_dir=results_dir, episode=episode)

    drone_names = sorted(full_drone_positions)

    start_idx, end_idx = offset, offset + window
    _warn_if_window_clamped(start_idx, end_idx, target_positions.shape[0])

    drone_positions = {name: pos[start_idx:end_idx] for name, pos in full_drone_positions.items()}
    target_positions_by_drone = {
        name: pos[start_idx:end_idx] for name, pos in full_target_positions_by_drone.items()
    }
    target_positions = target_positions[start_idx:end_idx]
    box_positions = box_positions[start_idx:end_idx]
    beliefs = {
        observer: {target: pos[start_idx:end_idx] for target, pos in targets.items()}
        for observer, targets in full_beliefs.items()
    }

    n_steps = target_positions.shape[0]
    fade = np.linspace(0.1, 0.9, n_steps)

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
    ax_left = fig.add_subplot(gs[:, 0], projection='3d')
    right_axes = [fig.add_subplot(gs[i, 1]) for i in range(len(drone_names))]
    error_axes = [fig.add_subplot(gs[i, 2]) for i in range(len(drone_names))]

    ax_left.set_facecolor('white')
    ax_left.computed_zorder = False
    ax_left.view_init(elev=CAMERA_ELEV, azim=CAMERA_AZIM)

    legend_handles = _draw_ground_truth(ax_left, drone_positions, target_positions, box_positions, fade)

    center_box = box_positions[-1][box_positions.shape[1] // 2]
    # this panel's rect (set below) has a different width:height ratio than
    # the standalone plot CAMERA_ZOOM was tuned against, so it gets its own
    # empirically-checked zoom (see the fill/clipping sweep this was tuned
    # with in the conversation that added this function) rather than reusing
    # CAMERA_ZOOM and under-filling the column
    _set_axes_bounds(ax_left, center_box, zoom=2.2)
    _strip_grid_and_ticks(ax_left)
    ax_left.set_title(
        f'Episode Ground Truth (steps {start_idx}-{start_idx + n_steps - 1})', fontweight='bold', fontsize=14,
    )
    ax_left.legend(handles=legend_handles, loc='upper right', framealpha=0.9, prop={'weight': 'bold', 'size': 12})

    full_step_axis = np.arange(full_drone_positions[drone_names[0]].shape[0])

    for ax, error_ax, observer in zip(right_axes, error_axes, drone_names):
        observer_target_positions = target_positions_by_drone[observer]
        _draw_ground_truth_2d(
            ax, estimated_positions_by_observer[observer], observer_target_positions, box_positions, fade,
            trail_size=8, final_size=45,
        )
        _set_axes_bounds_2d(ax, center_box[:2])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{observer.capitalize()} Belief', fontweight='bold', fontsize=11)

        # belief error per teammate over the FULL episode (not just the
        # window plotted on the left/middle panels), same color as that
        # teammate's markers
        for target_name in drone_names:
            if target_name == observer:
                continue
            color = DRONE_COLORS[drone_names.index(target_name) % len(DRONE_COLORS)]
            error = np.linalg.norm(
                full_drone_positions[target_name] - full_beliefs[observer][target_name], axis=1,
            )

            error_ax.plot(full_step_axis, error, color=color, linewidth=1.5, label=target_name)

        error_ax.axvspan(start_idx, end_idx - 1, color='grey', alpha=0.15, zorder=0)
        error_ax.set_ylim(bottom=0)
        error_ax.set_ylabel('belief error')
        error_ax.set_title(f'{observer.capitalize()} Error', fontweight='bold', fontsize=11)
        error_ax.legend(prop={'weight': 'bold', 'size': 8})

    error_axes[-1].set_xlabel('step')

    fig.tight_layout()
    # tight_layout lays every axes out including ax_left, but the 3d axes'
    # zoom/fill depends on overriding its position afterward (see
    # plot_episode_trajectory_overlay) - this rect and the zoom=2.2 above
    # were tuned together against this figure's left column specifically
    ax_left.set_position((0.01, 0.05, 0.36, 0.85))
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
