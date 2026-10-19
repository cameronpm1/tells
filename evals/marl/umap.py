# similarity to fully-observable: overall mean nearest-baseline distance in raw
# obs space (report_nn_to_baseline, lower = closer), all episodes,
# --n-neighbors 15 --min-dist 0.1 (logs in logs/umap_runs/)
#   drones        (results/drones, 30-d):        allocentric 1.0644  pf 2.0723
#   football      (results/football, 26-d):      allocentric 0.1603  pf 1.0094
#   predator_prey (predator_prey_fully_obs/checkpoint1000, 15-d): allocentric 1.9783  pf 2.3129
#   fire          (results/fire, umap_fire.py: checkpoint5000 policy conv features, 128-d, not raw pixels):
#                                                allocentric 1.5259  pf 1.6057

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# this file is itself named umap.py, so on a direct `python evals/marl/umap.py`
# invocation python auto-inserts this file's own directory at sys.path[0] -
# `import umap` below would then import this module instead of the installed
# umap-learn package. drop any sys.path entry that resolves to this file's
# directory before importing, so umap-learn always wins regardless of how
# this script is invoked (direct script, `python -m evals.marl.umap`, ...)
_here = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here]

import umap  # noqa: E402  (umap-learn, not this file - see guard above)
from sklearn.neighbors import NearestNeighbors  # noqa: E402


DEFAULT_SAVE_PATH = '/home/cameron/tells/test_umap.png'

# subdirectories looked for under the given checkpoint directory
SOURCE_DIRNAMES = {
    'true': 'results_fully_obs',
    'allocentric': 'results_allocentric',
    'pf': 'results_pf',
}
SOURCE_LABELS = {
    'true': 'baseline (obs)',
    'allocentric': 'allocentric NN (obs)',
    'pf': 'particle filter (obs)',
}
SOURCE_COLORS = {'true': '#4646aa', 'allocentric': '#aa4646', 'pf': '#32aa32'}


def list_episodes(results_dir: str):
    '''
    return the sorted episode indices for the <idx>.npz result files found
    directly inside results_dir
    '''
    paths = glob.glob(os.path.join(results_dir, '*.npz'))
    return sorted(int(os.path.splitext(os.path.basename(p))[0]) for p in paths)


def _select_episodes(results_dir: str, episodes, max_episodes, seed: int):
    all_episodes = list_episodes(results_dir)
    if not all_episodes:
        raise ValueError(f'no <idx>.npz result files found in {results_dir}')

    if episodes is not None:
        wanted = set(episodes)
        all_episodes = [e for e in all_episodes if e in wanted]
        if not all_episodes:
            raise ValueError(f'none of the requested episodes {sorted(wanted)} were found in {results_dir}')

    if max_episodes is not None and len(all_episodes) > max_episodes:
        rng = np.random.default_rng(seed)
        all_episodes = sorted(rng.choice(all_episodes, size=max_episodes, replace=False).tolist())

    return all_episodes


def _iter_episode_obs(npz_path: str):
    '''yield (step:int, obs:dict) for every step in one episode npz'''
    data = np.load(npz_path, allow_pickle=True)
    steps = sorted((f for f in data.files if f != 'obs_map'), key=int)
    if not steps:
        raise ValueError(f'no steps found in {npz_path}')

    for step in steps:
        obs, _rewards, _terminations, _truncations, _infos = data[step]
        yield int(step), obs


def load_obs_vectors(results_dir: str, episodes=None, max_episodes=None, seed: int = 0):
    '''
    the actual per-agent observation array saved at every (step, agent) in
    results_dir - i.e. obs, the same array the policy was fed that step
    (evals/marl/eval.py's eval_single_episode: results[step] = [obs,
    rewards, terminations, truncations, infos]) - not just the belief
    model's team-slice prediction, so this includes every other field in
    the observation too (self position, target/adversary position and
    goal, etc.)

    for the baseline 'results' directory (collected with no belief model)
    obs is the true, fully-observable reading. for 'results_allocentric'/
    'results_pf', obs has already had its team slice overwritten with that
    belief model's estimate (see RLLibWrapper.step / PFWrapper.step) while
    every other field (self/target/goal) stays the real, observed value -
    so this comparison reflects the whole input the policy actually saw,
    not just the belief model's own prediction accuracy

    per-agent obs that's already a flat vector (predator_prey, football,
    drones) is used as-is; a multi-dim array (fire's (61, 61, 1) fire-grid
    image) is flattened first, since UMAP - like any classical
    dimensionality reduction - only operates on (n_samples, n_features)
    tabular input, not images. fire's grid is a small categorical occupancy
    map (see observation_packaging.py's _one_hot_fire_grid), not a natural
    photo, so raw-pixel flattening is a reasonable representation here
    rather than needing a learned encoder first

    returns
    -------
    vectors:(N, D) array, agent_labels:(N,) array, episode_labels:(N,) int
    array, step_labels:(N,) int array
    '''
    episode_ids = _select_episodes(results_dir, episodes, max_episodes, seed)

    rows, agent_labels, episode_labels, step_labels = [], [], [], []
    for episode in episode_ids:
        npz_path = os.path.join(results_dir, f'{episode}.npz')
        for step, obs in _iter_episode_obs(npz_path):
            for agent, agent_obs in obs.items():
                if not isinstance(agent_obs, np.ndarray):
                    raise ValueError(
                        f"{npz_path}: obs[{agent!r}] is a {type(agent_obs).__name__}, not an array - "
                        "this script only supports array-valued per-agent obs"
                    )
                rows.append(np.asarray(agent_obs, dtype=float).ravel())
                agent_labels.append(agent)
                episode_labels.append(episode)
                step_labels.append(step)

    if not rows:
        raise ValueError(f'no usable (step, agent) rows found in {results_dir}')

    return (
        np.stack(rows),
        np.array(agent_labels),
        np.array(episode_labels, dtype=int),
        np.array(step_labels, dtype=int),
    )


def subsample_indices(n_total: int, max_points, seed: int):
    if max_points is None or n_total <= max_points:
        return np.arange(n_total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=max_points, replace=False))


def report_nn_to_baseline(source: str, belief_vecs, belief_agents, true_vecs, true_agents):
    '''
    for each agent, find the nearest baseline (true) point to every belief
    point in the raw D-dim observation space (not the UMAP embedding, whose
    distances aren't metrically meaningful) - unlike a single eval run's
    row-aligned belief error, the baseline/allocentric/pf directories are
    independent eval rollouts of different lengths, so there's no
    step-for-step pairing left to compare; nearest-neighbor distance is the
    alignment-free stand-in
    '''
    agents = sorted(set(belief_agents) & set(true_agents))
    if not agents:
        print(f'{source}: no agents overlap with the baseline, skipping distance report')
        return

    print(f'{source} -> nearest baseline point, raw {belief_vecs.shape[1]}-d observation:')
    all_dist = []
    for agent in agents:
        reference = true_vecs[true_agents == agent]
        query = belief_vecs[belief_agents == agent]
        nn = NearestNeighbors(n_neighbors=1).fit(reference)
        dist, _ = nn.kneighbors(query)
        dist = dist.ravel()
        all_dist.append(dist)
        print(f'  {agent}: mean {dist.mean():.4f}  median {np.median(dist):.4f}  n={len(dist)}')

    all_dist = np.concatenate(all_dist)
    print(f'  overall: mean {all_dist.mean():.4f}  median {np.median(all_dist):.4f}  std {all_dist.std():.4f}')


def run_umap(vectors_by_source: dict, n_neighbors: int, min_dist: float, metric: str, seed: int):
    '''
    fit a single UMAP reducer across every source's vectors concatenated
    together, so all clusters land in the same 2D coordinate space and can
    be compared directly on one plot - fitting a separate reducer per
    source would each pick its own arbitrary embedding, making the overlay
    meaningless

    returns
    -------
    dict: source -> (N_source, 2) embedding, fitted reducer
    '''
    order = list(vectors_by_source.keys())
    combined = np.concatenate([vectors_by_source[k] for k in order], axis=0)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=seed)
    embedding = reducer.fit_transform(combined)

    embeddings_by_source = {}
    offset = 0
    for k in order:
        n = len(vectors_by_source[k])
        embeddings_by_source[k] = embedding[offset:offset + n]
        offset += n

    return embeddings_by_source, reducer


def plot_embeddings(embeddings_by_source: dict, save_path: str, title: str, bins: int = 200):
    '''
    one row, one column per source (true/allocentric/pf, whichever are
    present), each panel a 2D histogram of that source's point density
    rather than a raw scatter - with UMAP fit over every collected point by
    default (see --max-points), a scatter plot is mostly overplotted blobs;
    density makes the actual shape of each source's manifold, and where
    it's thin vs. concentrated, legible

    every panel shares the same bin edges, axis limits, and color scale
    (all computed from the union of all sources) so a given position and a
    given color both mean the same thing across panels, making the
    comparison direct rather than needing to eyeball differently-scaled plots.
    empty bins are left white (not the colormap's zero value) with darker
    green marking higher point density
    '''
    sources = list(embeddings_by_source.keys())
    all_points = np.concatenate([embeddings_by_source[s] for s in sources], axis=0)

    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    x_pad = 0.05 * (x_max - x_min or 1.0)
    y_pad = 0.05 * (y_max - y_min or 1.0)
    x_edges = np.linspace(x_min - x_pad, x_max + x_pad, bins + 1)
    y_edges = np.linspace(y_min - y_pad, y_max + y_pad, bins + 1)

    histograms = {
        source: np.histogram2d(embeddings_by_source[source][:, 0], embeddings_by_source[source][:, 1],
                                bins=[x_edges, y_edges])[0]
        for source in sources
    }
    vmax = max(h.max() for h in histograms.values())

    fig, axes = plt.subplots(1, len(sources), figsize=(5 * len(sources), 5), sharex=True, sharey=True,
                              constrained_layout=True)
    if len(sources) == 1:
        axes = [axes]

    mesh = None
    for ax, source in zip(axes, sources):
        ax.set_facecolor('white')
        # mask empty bins so they show the axes' white background instead of
        # being painted the colormap's zero-density color - only bins that
        # actually contain points get shaded, light green -> dark green as
        # density increases
        masked = np.ma.masked_equal(histograms[source].T, 0)
        # log-scale the color norm: a linear scale from 0 to the peak count
        # leaves most bins (which sit far below that peak) looking nearly
        # white, since a handful of hotspots dominate the range - log
        # compresses that range so the bulk of the manifold reads as visibly
        # green instead of only the hottest few cells
        mesh = ax.pcolormesh(x_edges, y_edges, masked, cmap='Greens', norm=LogNorm(vmin=1, vmax=vmax))
        ax.set_title(SOURCE_LABELS.get(source, source))
        ax.set_xlabel('UMAP-1')
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])

    axes[0].set_ylabel('UMAP-2')
    fig.colorbar(mesh, ax=axes, label='point count (log scale)', fraction=0.02, pad=0.02)
    fig.suptitle(title)
    fig.savefig(save_path, dpi=300)
    print(f'saved plot to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "UMAP comparison of a checkpoint's baseline observation against the observations actually "
            "seen under its belief models. Looks under checkpoint_dir for up to three subdirectories of "
            "<idx>.npz eval results - 'results' (baseline, no belief model), 'results_allocentric' and "
            "'results_pf' - and plots each source's full per-agent obs array (not just the belief "
            "model's team-slice prediction) as its own cluster in one shared 2D embedding"
        )
    )
    parser.add_argument('checkpoint_dir', type=str,
                         help='checkpoint directory containing results/, results_allocentric/, '
                              'results_pf/ subdirectories, e.g. logs/marl/predator_prey_fully_obs/checkpoint1000')
    parser.add_argument('--episodes', type=str, default=None,
                         help='comma-separated episode indices to include from each source directory '
                              '(default: all found in that directory)')
    parser.add_argument('--max-episodes', type=int, default=None,
                         help='randomly subsample to at most this many episode files per source directory '
                              'before loading (default: use all found episodes)')
    parser.add_argument('--max-points', type=int, default=0,
                         help='randomly subsample each source to at most this many (step, agent) points '
                              'before fitting UMAP (default: 0, i.e. disabled - UMAP is fit over every '
                              'point collected; set this for a faster, subsampled fit on large results dirs)')
    parser.add_argument('--n-neighbors', type=int, default=15)
    parser.add_argument('--min-dist', type=float, default=0.1)
    parser.add_argument('--metric', type=str, default='euclidean')
    parser.add_argument('--bins', type=int, default=200,
                         help='number of density histogram bins per axis, shared across all panels (default: 200)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-path', type=str, default=DEFAULT_SAVE_PATH)
    args = parser.parse_args()

    episodes = None
    if args.episodes is not None:
        episodes = {int(e) for e in args.episodes.split(',')}
    max_points = args.max_points if args.max_points > 0 else None

    true_dir = os.path.join(args.checkpoint_dir, SOURCE_DIRNAMES['true'])
    if not os.path.isdir(true_dir):
        raise SystemExit(
            f"no baseline '{SOURCE_DIRNAMES['true']}' directory found at {true_dir} - this script needs it "
            "as the true-observation reference cluster (evals/marl/eval.py run with collect_results=True)"
        )

    vectors_by_source, agents_by_source, episodes_by_source = {}, {}, {}

    true_vecs, true_agents, true_episodes, _true_steps = load_obs_vectors(
        true_dir, episodes=episodes, max_episodes=args.max_episodes, seed=args.seed,
    )
    vectors_by_source['true'] = true_vecs
    agents_by_source['true'] = true_agents
    episodes_by_source['true'] = true_episodes
    print(f"true: loaded {len(true_vecs)} points from {len(set(true_episodes.tolist()))} episodes in {true_dir}")

    for source in ('allocentric', 'pf'):
        source_dir = os.path.join(args.checkpoint_dir, SOURCE_DIRNAMES[source])
        if not os.path.isdir(source_dir):
            print(f"{source}: no '{SOURCE_DIRNAMES[source]}' directory found at {source_dir}, skipping")
            continue

        belief_vecs, belief_agents, belief_episodes, _belief_steps = load_obs_vectors(
            source_dir, episodes=episodes, max_episodes=args.max_episodes, seed=args.seed,
        )
        vectors_by_source[source] = belief_vecs
        agents_by_source[source] = belief_agents
        episodes_by_source[source] = belief_episodes
        print(f"{source}: loaded {len(belief_vecs)} points from "
              f"{len(set(belief_episodes.tolist()))} episodes in {source_dir}")

    if len(vectors_by_source) < 2:
        raise SystemExit(
            f"neither '{SOURCE_DIRNAMES['allocentric']}' nor '{SOURCE_DIRNAMES['pf']}' was found under "
            f"{args.checkpoint_dir} - nothing to compare against the baseline"
        )

    for source in ('allocentric', 'pf'):
        if source in vectors_by_source:
            report_nn_to_baseline(source, vectors_by_source[source], agents_by_source[source], true_vecs, true_agents)

    for source, vectors in vectors_by_source.items():
        idx = subsample_indices(len(vectors), max_points, args.seed)
        if len(idx) < len(vectors):
            print(f'{source}: subsampled to {len(idx)} points for the UMAP fit (--max-points={args.max_points})')
        vectors_by_source[source] = vectors[idx]

    embeddings_by_source, _reducer = run_umap(
        vectors_by_source, n_neighbors=args.n_neighbors, min_dist=args.min_dist, metric=args.metric, seed=args.seed,
    )

    plot_embeddings(
        embeddings_by_source,
        save_path=args.save_path,
        title=f'UMAP: baseline vs belief observations\n{args.checkpoint_dir}',
        bins=args.bins,
    )
