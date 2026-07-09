import itertools
import numpy as np

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
        9: np.array([0, 0]),     #extinguish fire
    }

def _compute_drone_slots(
    obs,
    fire_value=1,
    green_value=0,
    black_value=2,
    n_slots=4,
    min_slot_dist=6,
    preferred_band=(2.0, 3.0),
):
    agents = [a for a in obs if a != "target"]

    if "target" in obs and "target" in obs["target"]:
        fire = np.asarray(obs["target"])
        red_cells = np.argwhere(fire == fire_value).astype(float)
        green_cells = np.argwhere(fire == green_value).astype(float)
        black_cells = np.argwhere(fire == black_value).astype(float)
    else:
        red_cells, green_cells, black_cells = [], [], []
        for a in agents:
            local_fire = np.asarray(obs[a]["fire"])
            pos = np.asarray(obs[a]["pos"], dtype=float)
            radius = np.array(local_fire.shape) // 2

            for value, store in [
                (fire_value, red_cells),
                (green_value, green_cells),
                (black_value, black_cells),
            ]:
                cells = np.argwhere(local_fire == value)
                if len(cells):
                    store.append(cells + pos - radius)

        red_cells = np.vstack(red_cells).astype(float) if red_cells else np.empty((0, 2))
        green_cells = np.vstack(green_cells).astype(float) if green_cells else np.empty((0, 2))
        black_cells = np.vstack(black_cells).astype(float) if black_cells else np.empty((0, 2))

    if len(red_cells) == 0:
        return np.array([obs[a]["pos"] for a in agents], dtype=float)

    red_cells = np.unique(np.round(red_cells).astype(int), axis=0).astype(float)
    green_cells = np.unique(np.round(green_cells).astype(int), axis=0).astype(float)
    black_cells = np.unique(np.round(black_cells).astype(int), axis=0).astype(float)

    def min_dist(points, refs):
        if len(refs) == 0:
            return np.full(len(points), np.inf)
        return np.min(np.linalg.norm(points[:, None, :] - refs[None, :, :], axis=2), axis=1)

    def inward_band_cost(d):
        lo, hi = preferred_band
        return np.where(
            d < lo,
            1000.0 + (lo - d),      # strongly avoid immediate edge cells
            np.maximum(0.0, d - hi) # after 2-3 px, move inward gradually
        )

    d_green = min_dist(red_cells, green_cells)
    priority = inward_band_cost(d_green)

    if len(black_cells) > 0:
        d_black = min_dist(red_cells, black_cells)
        priority += inward_band_cost(d_black)

    drone_center = np.mean([obs[a]["pos"] for a in agents], axis=0)
    priority += 0.01 * np.linalg.norm(red_cells - drone_center, axis=1)

    order = np.argsort(priority)
    slots = []

    for idx in order:
        cell = red_cells[idx]
        if all(np.linalg.norm(cell - s) >= min_slot_dist for s in slots):
            slots.append(cell)
        if len(slots) == n_slots:
            break

    for idx in order:
        if len(slots) == n_slots:
            break
        cell = red_cells[idx]
        if not any(np.array_equal(cell, s) for s in slots):
            slots.append(cell)

    while len(slots) < len(agents):
        slots.append(slots[-1])

    return np.array(slots[:len(agents)], dtype=float)


def extinguish_controller(
        obs, 
        fire_value=1, 
        extinguish_radius=5, 
        min_slot_dist=6
    ):
    agents = [a for a in obs if a != "target"]
    vec_to_action = {tuple(v): k for k, v in ACTION_LIBRARY.items() if k != 9}

    slots = _compute_drone_slots(
        obs,
        fire_value=fire_value,
        n_slots=len(agents),
        min_slot_dist=min_slot_dist,
    )

    positions = np.array([obs[a]["pos"] for a in agents], dtype=float)

    assignment = min(
        itertools.permutations(range(len(slots))),
        key=lambda p: sum(np.linalg.norm(positions[i] - slots[p[i]]) for i in range(len(agents))),
    )

    actions = {}

    for i, agent in enumerate(agents):
        pos = positions[i]
        slot = slots[assignment[i]]

        if np.linalg.norm(slot - pos) <= extinguish_radius:
            actions[agent] = 9
        else:
            step = np.sign(slot - pos).astype(int)
            actions[agent] = vec_to_action.get(tuple(step), 0)

    return actions

def probabilistic_fire_controller(
    fire_state,
    spread_prob=0.05,
    use_diagonals=True,
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
    """

    GREEN, RED, BLACK = 0, 1, 2

    new_fire = fire_state.copy()
    rows, cols = fire_state.shape

    if use_diagonals:
        neighbor_offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
    else:
        neighbor_offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
        ]

    burning_cells = np.argwhere(fire_state == RED)

    for r, c in burning_cells:
        for dr, dc in neighbor_offsets:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols:
                if fire_state[nr, nc] == GREEN:
                    if np.random.random() < spread_prob:
                        new_fire[nr, nc] = RED

    # Preserve black cells explicitly
    new_fire[fire_state == BLACK] = BLACK

    return new_fire