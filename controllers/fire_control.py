import itertools
import numpy as np
from scipy.spatial import cKDTree

ACTION_LIBRARY = {
        0: np.array([0, 0]),     # stay
        1: np.array([-1, 0]),    # up
        2: np.array([1, 0]),     # down
        3: np.array([0, -1]),    # left
        4: np.array([0, 1]),     # right
        5: np.array([-1, -1]),   # up-left
        6: np.array([-1, 1]),    # up-right
        7: np.array([1, -1]),    # down-left
        8: np.array([1, 1]),     # down-right
        9: None,     #extinguish fire
    }

def nearest_cell_dist(points, refs):
    if len(refs) == 0:
        return np.full(len(points), np.inf)
    dist, _ = cKDTree(refs).query(points, k=1)
    return dist


def inward_band_cost(d, preferred_band=(2.0, 3.0)):
    lo, hi = preferred_band
    return np.where(
        d < lo,
        1000.0 + (lo - d),      # strongly avoid immediate edge cells
        np.maximum(0.0, d - hi) # after 2-3 px, move inward gradually
    )


def _extract_cells(obs, value):
    agents = [a for a in obs if a != "target"]

    if "target" in obs:
        fire = np.asarray(obs["target"])
        cells = np.argwhere(fire == value).astype(float)
    else:
        cells = []
        for a in agents:
            local_fire = np.asarray(obs[a]["fire"])
            pos = np.asarray(obs[a]["self_pos"], dtype=float)
            radius = np.array(local_fire.shape) // 2
            local_cells = np.argwhere(local_fire == value)
            if len(local_cells):
                cells.append(local_cells + pos - radius)
        cells = np.vstack(cells).astype(float) if cells else np.empty((0, 2))

    if len(cells):
        cells = np.unique(np.round(cells).astype(int), axis=0).astype(float)

    return cells


def _compute_line_slots(
    red_cells,
    depth_cost,
    positions,
    n_slots,
    min_slot_dist=6.0,
    line_percentile=15.0,
):
    """
    Lays out `n_slots` points on a single straight line perpendicular to
    the axis from the drone cluster's center to the fire's center, all at
    the *same* depth into the fire (a low percentile of how far each red
    cell sits along that axis) - so the slots describe one line advancing
    as a unit rather than independent per-drone targets. Each slot snaps
    to a real red cell near its ideal position, preferring cells in the
    inward depth band (`depth_cost`) over ones right on the boundary. As
    the near edge is extinguished, this shared depth recedes into the
    fire on its own each time it's recomputed, so the whole line advances
    together.
    """
    drone_center = positions.mean(axis=0)
    fire_center = red_cells.mean(axis=0)

    approach = fire_center - drone_center
    approach_norm = np.linalg.norm(approach)
    approach_dir = approach / approach_norm if approach_norm > 1e-6 else np.array([1.0, 0.0])
    tangent_dir = np.array([-approach_dir[1], approach_dir[0]])

    proj_along = (red_cells - fire_center) @ approach_dir
    proj_tangent = (red_cells - fire_center) @ tangent_dir

    # one shared line depth for the whole team, not a per-drone value
    line_depth = np.percentile(proj_along, line_percentile)
    near_mask = proj_along <= line_depth + min_slot_dist
    if not np.any(near_mask):
        near_mask = np.ones(len(red_cells), dtype=bool)
    candidates = np.where(near_mask)[0]

    fire_width = (
        proj_tangent[candidates].max() - proj_tangent[candidates].min()
        if len(candidates) > 1 else 0.0
    )
    spacing = np.clip(
        fire_width / max(n_slots - 1, 1),
        min_slot_dist,
        min_slot_dist * 3.0,
    )
    tangent_center = proj_tangent[candidates].mean() if len(candidates) else 0.0
    ideal_offsets = (np.arange(n_slots) - (n_slots - 1) / 2.0) * spacing + tangent_center

    slots = []
    used = set()
    for target_offset in ideal_offsets:
        scores = np.abs(proj_tangent[candidates] - target_offset) + 2.0 * depth_cost[candidates]
        order = np.argsort(scores)
        chosen = None
        for idx in order:
            cell_idx = candidates[idx]
            if cell_idx not in used:
                used.add(cell_idx)
                chosen = red_cells[cell_idx]
                break
        slots.append(chosen if chosen is not None else fire_center)

    return np.array(slots, dtype=float)


def extinguish_controller(
        obs,
        fire_value=1,
        green_value=0,
        black_value=2,
        extinguish_radius=5,
        min_slot_dist=6,
        deadzone=0.0,
        preferred_band=(2.0, 3.0),
    ):
    """
    Drives the team as a single line perpendicular to the axis from the
    drone cluster's center to the fire's center, sweeping through from the
    side the team approached. All drones target the same shared depth into
    the fire each step (see `_compute_line_slots`), only their lateral
    position on the line differs, so the team advances in unison rather
    than racing ahead independently. Line positions snap to real fire
    cells chosen via `preferred_band` depth from the green/black boundary
    (like the old point-defense controller's goal cells) so the line sits
    inside the fire rather than skimming its edge.
    """
    agents = [a for a in obs if a != "target"]
    vec_to_action = {tuple(v): k for k, v in ACTION_LIBRARY.items() if k != 9}

    positions = np.array([obs[a]["self_pos"] for a in agents], dtype=float)
    red_cells = _extract_cells(obs, fire_value)

    if len(red_cells) == 0:
        return {agent: 0 for agent in agents}

    green_cells = _extract_cells(obs, green_value)
    black_cells = _extract_cells(obs, black_value)

    d_green = nearest_cell_dist(red_cells, green_cells)
    depth_cost = inward_band_cost(d_green, preferred_band)

    if len(black_cells) > 0:
        d_black = nearest_cell_dist(red_cells, black_cells)
        depth_cost = depth_cost + inward_band_cost(d_black, preferred_band)

    slots = _compute_line_slots(
        red_cells,
        depth_cost,
        positions,
        n_slots=len(agents),
        min_slot_dist=min_slot_dist,
    )

    # stable lateral assignment so the line doesn't cross itself between
    # calls: each drone keeps roughly the slot closest to where it is
    assignment = min(
        itertools.permutations(range(len(slots))),
        key=lambda p: sum(np.linalg.norm(positions[i] - slots[p[i]]) for i in range(len(agents))),
    )

    # extinguish based on real proximity to fire (not proximity to the
    # abstract slot) so a drone only holds position once it can actually
    # reach the flames, and a small extinguish_radius here (decoupled from
    # the simulator's actual extinguish blast radius) keeps the line
    # actively advancing instead of camping wherever ambient regrowth
    # keeps something nearby
    dist_to_fire = nearest_cell_dist(positions, red_cells)

    actions = {}
    for i, agent in enumerate(agents):
        pos = positions[i]
        slot = slots[assignment[i]]

        if dist_to_fire[i] <= extinguish_radius + deadzone:
            actions[agent] = 9
            continue

        if np.linalg.norm(slot - pos) <= extinguish_radius + deadzone:
            nearest_idx = np.argmin(np.linalg.norm(red_cells - pos, axis=1))
            target = red_cells[nearest_idx]
        else:
            target = slot

        step = np.sign(target - pos).astype(int)
        actions[agent] = vec_to_action.get(tuple(step), 0)

    return actions

def probabilistic_fire_controller(
    fire_state,
    drone_positions=None,
    spread_prob=0.05,
    use_diagonals=True,
    away_bias=1.0,
):
    """
    Probabilistically spreads fire outward from burning cells.

    Cell meanings:
        0 = GREEN
        1 = RED
        2 = BLACK

    Rules:
        - RED cells stay RED.
        - GREEN cells adjacent to RED cells may become RED.
        - BLACK cells remain BLACK and cannot reignite.

    When drone_positions is given, spread is anisotropic: growth in the
    direction away from the drone cluster's center is boosted and growth
    back toward the cluster is suppressed, so the fire front tends to flee
    the team.
    """

    GREEN, RED, BLACK = 0, 1, 2

    rows, cols = fire_state.shape
    red_mask = fire_state == RED
    green_mask = fire_state == GREEN

    if use_diagonals:
        neighbor_offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
    else:
        neighbor_offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
        ]

    # Pad by 1 so each neighbor offset can be read via a plain slice
    # (out-of-bounds neighbors read as False).
    padded_red = np.pad(red_mask, 1, mode="constant", constant_values=False)

    away_r = away_c = None
    if drone_positions is not None and len(drone_positions) > 0:
        cluster_center = np.mean(np.asarray(drone_positions, dtype=float), axis=0)
        rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        away_r = rr - cluster_center[0]
        away_c = cc - cluster_center[1]
        away_norm = np.hypot(away_r, away_c) + 1e-6
        away_r, away_c = away_r / away_norm, away_c / away_norm

    ignite = np.zeros_like(green_mask)
    for dr, dc in neighbor_offsets:
        neighbor_is_red = padded_red[1 + dr: 1 + dr + rows, 1 + dc: 1 + dc + cols]

        if away_r is not None:
            offset_norm = np.hypot(dr, dc)
            # a cell ignites from a red neighbor at (dr, dc); the growth
            # direction (source -> newly-igniting cell) is (-dr, -dc)
            alignment = (-dr * away_r - dc * away_c) / offset_norm
            multiplier = np.clip(1.0 + away_bias * alignment, 0.0, None)
            cell_spread_prob = np.clip(spread_prob * multiplier, 0.0, 1.0)
        else:
            cell_spread_prob = spread_prob

        trial = np.random.random((rows, cols)) < cell_spread_prob
        ignite |= neighbor_is_red & trial

    new_fire = fire_state.copy()
    new_fire[green_mask & ignite] = RED

    # Preserve black cells explicitly
    new_fire[fire_state == BLACK] = BLACK

    return new_fire
