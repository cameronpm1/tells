import numpy as np

def probabilistic_fire_controller(
    fire_state,
    spread_prob=0.07,
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