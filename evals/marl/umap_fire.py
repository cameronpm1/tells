import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
from ray.rllib.policy.policy import Policy

_here = os.path.dirname(os.path.abspath(__file__))

# the pickled policy state under checkpoint_dir/policies/<id>/policy_state.pkl
# references this repo's own modules (e.g. learn.marl.*) by import path, so
# unpickling it via Policy.from_checkpoint needs the repo root importable -
# not guaranteed by default when running this script directly (python puts
# this file's own directory, evals/marl, on sys.path[0], not the repo root)
_repo_root = os.path.dirname(os.path.dirname(_here))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# umap.py (this directory) shares its name with the umap-learn package, so a
# plain `import umap` here would be ambiguous - load it explicitly by file
# path instead, under a private name, to reuse its episode-loading/
# reporting/plotting helpers without fighting that name collision
_umap_spec = importlib.util.spec_from_file_location('_umap_common', os.path.join(_here, 'umap.py'))
_umap_common = importlib.util.module_from_spec(_umap_spec)
_umap_spec.loader.exec_module(_umap_common)


DEFAULT_SAVE_PATH = '/home/cameron/tells/test_fire_umap_latent.png'


def load_policy_model(checkpoint_dir: str, policy_id: str = None):
    '''
    load the trained fire policy's torch model from an RLlib checkpoint
    directory (e.g. logs/marl/fire_fully_obs/checkpoint5000 - the same
    directory results/results_allocentric/results_pf live under)

    fire trains a single shared policy for every learned agent (see
    learn/marl/train.py's marl_policy_mapping_fn), stored under
    checkpoint_dir/policies/<policy_id>/ - if policy_id isn't given, this
    picks the one policy directory found there, or 'agent0' if there are
    several

    returns
    -------
    model (torch.nn.Module, eval mode), resolved policy_id
    '''
    if policy_id is None:
        policies_dir = os.path.join(checkpoint_dir, 'policies')
        available = sorted(os.listdir(policies_dir)) if os.path.isdir(policies_dir) else []
        if len(available) == 1:
            policy_id = available[0]
        elif 'agent0' in available:
            policy_id = 'agent0'
        else:
            raise ValueError(
                f"couldn't determine which policy to load from {policies_dir} (found {available}) - "
                "pass --policy-id explicitly"
            )

    # rllib parses checkpoint_dir as a URI, so a relative path fails with "empty scheme"
    policies = Policy.from_checkpoint(os.path.abspath(checkpoint_dir), policy_ids=[policy_id])
    model = policies[policy_id].model
    model.eval()
    return model, policy_id


def build_flatten_encoder(model):
    '''
    slice out just the conv stack + Flatten from RLlib's VisionNetwork
    (ray/rllib/models/torch/visionnet.py) - there's no separate named
    "CNN encoder" attribute on this model: model._convs fuses the conv
    layers, the Flatten, AND the post_fcnet_hiddens FC stack (everything
    but the final action-logits layer, which gets split out separately
    into model._logits) into one nn.Sequential. inspecting a loaded
    checkpoint's _convs shows the Flatten module sits right after the last
    conv layer and before the first FC layer, so slicing up to and
    including it gives exactly the first flattened representation of the
    image, before any of the FC layers see it
    '''
    modules = list(model._convs.children())
    flatten_idx = next(i for i, m in enumerate(modules) if isinstance(m, torch.nn.Flatten))
    return torch.nn.Sequential(*modules[:flatten_idx + 1])


def load_latent_vectors(results_dir, encoder, device, episodes=None, max_episodes=None, seed: int = 0,
                         batch_size: int = 1024):
    '''
    like umap.py's load_obs_vectors, but instead of using each agent's raw
    (61,61,1) fire-grid pixels directly, runs it through `encoder` (the
    trained policy's own conv stack up to its first flatten, see
    build_flatten_encoder) and uses the resulting feature vector as the
    point - the policy's own learned representation of the image, rather
    than raw pixels

    returns
    -------
    vectors:(N, D) array, agent_labels:(N,) array, episode_labels:(N,) int
    array, step_labels:(N,) int array
    '''
    episode_ids = _umap_common._select_episodes(results_dir, episodes, max_episodes, seed)

    raw_obs, agent_labels, episode_labels, step_labels = [], [], [], []
    for episode in episode_ids:
        npz_path = os.path.join(results_dir, f'{episode}.npz')
        for step, obs in _umap_common._iter_episode_obs(npz_path):
            for agent, agent_obs in obs.items():
                raw_obs.append(np.asarray(agent_obs, dtype=np.float32))
                agent_labels.append(agent)
                episode_labels.append(episode)
                step_labels.append(step)

    if not raw_obs:
        raise ValueError(f'no usable (step, agent) rows found in {results_dir}')

    latents = []
    with torch.no_grad():
        for start in range(0, len(raw_obs), batch_size):
            batch = np.stack(raw_obs[start:start + batch_size])
            features = torch.from_numpy(batch).to(device).permute(0, 3, 1, 2)
            latents.append(encoder(features).cpu().numpy())

    return (
        np.concatenate(latents, axis=0).astype(float),
        np.array(agent_labels),
        np.array(episode_labels, dtype=int),
        np.array(step_labels, dtype=int),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "UMAP comparison of the fire policy's own learned image representation, not raw pixels: loads "
            "the trained VisionNetwork from checkpoint_dir's RLlib checkpoint, runs every saved fire-grid "
            "observation through its conv stack up to the first flattened layer, and does the same "
            "baseline/allocentric/pf UMAP density comparison as umap.py but on that feature vector"
        )
    )
    parser.add_argument('checkpoint_dir', type=str,
                         help='checkpoint directory containing policies/, results/, results_allocentric/, '
                              'results_pf/ subdirectories, e.g. logs/marl/fire_fully_obs/checkpoint5000')
    parser.add_argument('--results-dir', type=str, default=None,
                         help='directory holding the results_* subdirectories, if not checkpoint_dir itself '
                              '(e.g. results/fire)')
    parser.add_argument('--policy-id', type=str, default=None,
                         help='which policy to load from checkpoint_dir/policies/ (default: auto-detect)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=1024,
                         help='batch size for running observations through the encoder (default: 1024)')
    parser.add_argument('--episodes', type=str, default=None,
                         help='comma-separated episode indices to include from each source directory '
                              '(default: all found in that directory)')
    parser.add_argument('--max-episodes', type=int, default=None,
                         help='randomly subsample to at most this many episode files per source directory '
                              'before loading (default: use all found episodes)')
    parser.add_argument('--max-points', type=int, default=0,
                         help='randomly subsample each source to at most this many (step, agent) points '
                              'before fitting UMAP (default: 0, i.e. disabled)')
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

    model, resolved_policy_id = load_policy_model(args.checkpoint_dir, policy_id=args.policy_id)
    model.to(args.device)
    encoder = build_flatten_encoder(model)

    with torch.no_grad():
        dummy = torch.zeros(1, *model.obs_space.shape).to(args.device).permute(0, 3, 1, 2)
        latent_dim = encoder(dummy).shape[1]
    print(f"loaded policy '{resolved_policy_id}' from {args.checkpoint_dir} on {args.device}, "
          f"flattened conv feature dim: {latent_dim}")

    results_root = args.results_dir or args.checkpoint_dir
    true_dir = os.path.join(results_root, _umap_common.SOURCE_DIRNAMES['true'])
    if not os.path.isdir(true_dir):
        raise SystemExit(
            f"no baseline '{_umap_common.SOURCE_DIRNAMES['true']}' directory found at {true_dir} - this script "
            "needs it as the true-observation reference cluster (evals/marl/eval.py run with collect_results=True)"
        )

    vectors_by_source, agents_by_source = {}, {}

    true_vecs, true_agents, true_episodes, _true_steps = load_latent_vectors(
        true_dir, encoder, args.device, episodes=episodes, max_episodes=args.max_episodes,
        seed=args.seed, batch_size=args.batch_size,
    )
    vectors_by_source['true'] = true_vecs
    agents_by_source['true'] = true_agents
    print(f"true: loaded {len(true_vecs)} points from {len(set(true_episodes.tolist()))} episodes in {true_dir}")

    for source in ('allocentric', 'pf'):
        source_dir = os.path.join(results_root, _umap_common.SOURCE_DIRNAMES[source])
        if not os.path.isdir(source_dir):
            print(f"{source}: no '{_umap_common.SOURCE_DIRNAMES[source]}' directory found at {source_dir}, skipping")
            continue

        belief_vecs, belief_agents, belief_episodes, _belief_steps = load_latent_vectors(
            source_dir, encoder, args.device, episodes=episodes, max_episodes=args.max_episodes,
            seed=args.seed, batch_size=args.batch_size,
        )
        vectors_by_source[source] = belief_vecs
        agents_by_source[source] = belief_agents
        print(f"{source}: loaded {len(belief_vecs)} points from "
              f"{len(set(belief_episodes.tolist()))} episodes in {source_dir}")

    if len(vectors_by_source) < 2:
        raise SystemExit(
            f"neither '{_umap_common.SOURCE_DIRNAMES['allocentric']}' nor '{_umap_common.SOURCE_DIRNAMES['pf']}' "
            f"was found under {args.checkpoint_dir} - nothing to compare against the baseline"
        )

    for source in ('allocentric', 'pf'):
        if source in vectors_by_source:
            _umap_common.report_nn_to_baseline(
                source, vectors_by_source[source], agents_by_source[source], true_vecs, true_agents,
            )

    for source, vectors in vectors_by_source.items():
        idx = _umap_common.subsample_indices(len(vectors), max_points, args.seed)
        if len(idx) < len(vectors):
            print(f'{source}: subsampled to {len(idx)} points for the UMAP fit (--max-points={args.max_points})')
        vectors_by_source[source] = vectors[idx]

    embeddings_by_source, _reducer = _umap_common.run_umap(
        vectors_by_source, n_neighbors=args.n_neighbors, min_dist=args.min_dist, metric=args.metric, seed=args.seed,
    )

    _umap_common.plot_embeddings(
        embeddings_by_source,
        save_path=args.save_path,
        title=(f"UMAP: baseline vs belief fire-policy conv features (policy '{resolved_policy_id}', "
               f"{latent_dim}-d)\n{args.checkpoint_dir}"),
        bins=args.bins,
    )
