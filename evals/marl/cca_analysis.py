"""Fixed-axis CCA comparison of belief models against the true state (see CCA_ANALYSIS.md).

Every method (fully observed, allocentric, PF) is scored on the SAME axes, fixed by the true model:
    task axes       the controller's task directions: CCA of the true state with the controller
                    targets on the fully-observed data (sigma^2 > --min-eig), own state regressed
                    out, applied to every method as the same linear functionals of the state
    physical axes   teammate centroid, teammate spread, adversary and any extra entity (football: ball)

Per axis and method, raw R^2 = 1 - MSE / var(truth) of the belief's own estimate (no fitting) - exactly
what the policy receives. Coverage = mean R^2 over the task directions. CIs from an episode bootstrap.

Variables per agent-step (all agents pooled, first --max-steps steps of each episode):
    z      true teammates, adversary and extras relative to the agent (obs_no_noise)
    z_hat  belief teammates (sampled_predictions), observed adversary and extras (PF: filter estimates)
    own    the agent's own true state, regressed out of the task CCA
    y      every agent's continuous controller targets on the true state, before discretization

Usage:
    python evals/marl/cca_analysis.py --env football --out-dir results/football/cca_final
    python plotting/plot_cca_analysis.py --in-dir results/football/cca_final
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_v, '24')

import argparse
import glob
import json
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
import controllers.drone_control as dc
import controllers.predator_prey_control as ppc

# compute_slot_actions discretizes (slot - pos) with vec_to_action; swapping in the identity returns
# the continuous predator-prey controller targets instead
ppc.vec_to_action = lambda vec, deadzone=0.05: np.asarray(vec, dtype=np.float64)

METHODS = ['fully_obs', 'allocentric', 'pf']  # fully_obs is the controller reference
CHECKPOINT_FOLDERS = {'fully_obs': 'results', 'allocentric': 'results_allocentric', 'pf': 'results_pf'}
SPEC = {}  # environment spec, filled in main, read by forked workers


# ---------------------------------------------------------------- controllers (continuous targets)

def pp_targets(obs, obs_map, agents, cfg):
    slots = ppc.compute_slot_actions({a: obs[a] for a in agents}, obs_map)
    return np.concatenate([slots[a] for a in agents])


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def football_targets(obs, obs_map, agents, cfg, closest_angle_2_target=0.6, expand_ratio=2.0):
    """desired_vec of controllers.football_control.compute_rondo_actions, before discretization.

    Mirrors its branches: the ball owner aims at its safest teammate, neighbours of the
    owner expand the circle, everyone else returns to their anchor.
    """
    out = []
    for i, a in enumerate(agents):
        ao = obs[a]
        pos = ao[obs_map['self_pos']]
        anchor = ao[obs_map['target_ball']][0:2]
        owner = np.where(ao[obs_map['ball_owner']] > 0)[0]
        owner = owner[0] if len(owner) else -1
        team_rel = ao[obs_map['team']].reshape(-1, 2)
        target_rel = ao[obs_map['target_pos']]
        if owner == i:
            angles = [abs(np.arctan2(_cross(target_rel, t), np.dot(target_rel, t))) for t in team_rel]
            idx = max(range(len(team_rel)), key=lambda j: angles[j] * int(angles[j] > closest_angle_2_target))
            vec = team_rel[idx]
        elif (i - owner) % 5 in (1, 5 - 1):
            vec = anchor * expand_ratio - pos
        else:
            vec = anchor - pos
        out.append(vec)
    return np.concatenate(out)


def drone_targets(obs, obs_map, agents, cfg):
    """move_vec of controllers.drone_control._compute_actions_from_state, before discretization."""
    _, boxes, adv = dc._parse_local_obs(obs[agents[0]], obs_map)
    prot = np.array([dc._parse_local_obs(obs[a], obs_map)[0] for a in agents], dtype=np.float64)
    slot_xy, perp = dc._desired_slot_coords(boxes.astype(np.float64), adv.astype(np.float64), cfg)
    slots = dc._slots_to_positions(slot_xy, float(cfg.get('base_altitude', np.mean(boxes[:, 2]))))
    order = np.argsort((prot[:, 0:2] - adv[0:2]) @ perp)
    vec = np.zeros_like(prot)
    for role, idx in enumerate(order):
        vec[idx] = slots[role] - prot[idx]
    return vec.ravel()


# ---------------------------------------------------------------- environment specs

def pp_success(last):
    return bool(last[4]['__common__']['agent0']['success'])


def football_success(last):  # kept possession until the step limit
    return bool(last[3]['__all__']) and not bool(last[2]['__all__'])


def drone_success(last):  # survived to the step limit without breach / out of bounds
    c0 = last[4]['__common__']['agent0']
    return bool(last[3]['__all__']) and not (c0['breached'] or c0['oob'])


ENVS = {
    'predator_prey': dict(
        agents=['agent0', 'agent1', 'agent2'], p=2, adversary='prey', own_slice=slice(0, 6),
        targets=pp_targets, config=None, success=pp_success, extras=[],
        checkpoint='logs/marl/predator_prey_fully_obs/checkpoint1000'),
    'football': dict(
        agents=['agent0', 'agent1', 'agent2', 'agent3', 'agent4'], p=2, adversary='opponent', own_slice=slice(0, 6),
        targets=football_targets, config=None, success=football_success, extras=[('ball', 'ball_pos')],
        checkpoint='logs/marl/football_fully_obs/checkpoint3000'),
    'drones': dict(  # controller targets use the fully-observed config's controller_kwargs for every method
        agents=['agent0', 'agent1', 'agent2'], p=3, adversary='intruder', own_slice=slice(0, 21),
        targets=drone_targets, config='confs/drones/3a_game.yaml', success=drone_success, extras=[],
        checkpoint='logs/marl/drones_fully_obs/checkpoint1500'),
}


def title():
    n_mates, dim = len(SPEC['agents']) - 1, SPEC['p']
    extras = [name for name, _ in SPEC['extras']]
    plus = ''.join(f' + {e}' for e in extras)
    return f"teammates + {SPEC['adversary']}{plus} ({(n_mates + 1 + len(extras)) * dim}-d)"


# ---------------------------------------------------------------- data

def load_episode(path, max_steps=None):
    agents = SPEC['agents']
    d = np.load(path, allow_pickle=True)
    obs_map = d['obs_map'].item()
    steps = sorted(int(k) for k in d.files if k.isdigit())[:max_steps]
    obs = {a: [] for a in agents}
    true = {a: [] for a in agents}
    pred = {a: [] for a in agents}
    y = []
    for k in steps:
        o, _, _, _, info = d[str(k)]
        common = info['__common__']
        nn = {a: np.asarray(common['obs_no_noise'][a], dtype=np.float64) for a in agents}
        sp = common.get('sampled_predictions', {})
        y.append(SPEC['targets'](nn, obs_map, agents, SPEC['cfg']))
        for a in agents:
            obs[a].append(np.asarray(o[a], dtype=np.float64))
            true[a].append(nn[a])
            pred[a].append(np.asarray(sp.get(a, o[a][obs_map['team']]), dtype=np.float64))
    y = np.array(y)
    out = {}
    for a in agents:
        ob, tr = np.array(obs[a]), np.array(true[a])
        out[a] = dict(team=tr[:, obs_map['team']], team_hat=np.array(pred[a]).reshape(len(steps), -1),
                      adv=tr[:, obs_map['target_pos']], adv_hat=ob[:, obs_map['target_pos']],
                      own=tr[:, SPEC['own_slice']], y=y)
        for name, key in SPEC['extras']:
            out[a][name] = tr[:, obs_map[key]]
            out[a][f'{name}_hat'] = ob[:, obs_map[key]]
    return out


def load_method(results_dir, workers, max_steps, limit=None):
    """Variable sets of one method, all agents pooled and ordered by episode."""
    files = sorted(glob.glob(os.path.join(results_dir, '*.npz')),
                   key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))[:limit]
    with Pool(workers) as p:
        eps = p.map(partial(load_episode, max_steps=max_steps), files)
    agents = SPEC['agents']
    per_agent = []
    for a in agents:
        da = {k: np.concatenate([e[a][k] for e in eps]) for k in eps[0][a]}
        da['ep'] = np.concatenate([np.full(len(e[a]['team']), i) for i, e in enumerate(eps)])
        per_agent.append(da)
    d = {k: np.concatenate([da[k] for da in per_agent]) for k in per_agent[0]}
    order = np.argsort(d['ep'], kind='stable')
    d = {k: v[order] for k, v in d.items()}
    extras = [name for name, _ in SPEC['extras']]
    out = dict(z=np.column_stack([d['team'], d['adv']] + [d[e] for e in extras]),
               zh=np.column_stack([d['team_hat'], d['adv_hat']] + [d[f'{e}_hat'] for e in extras]),
               own=d['own'], y=d['y'], adv=d['adv'], ep=d['ep'])
    out.update({e: d[e] for e in extras})
    return out


def episode_rows(ep):
    bounds = np.flatnonzero(np.diff(ep)) + 1
    return np.split(np.arange(len(ep)), bounds)


# ---------------------------------------------------------------- linear algebra

def inv_sqrt(S, eps=1e-6):
    lam, Q = np.linalg.eigh(S + eps * np.eye(len(S)))
    return Q @ np.diag(1.0 / np.sqrt(lam)) @ Q.T


def residualize(x, own):
    """Remove the linear dependence on the agent's own state."""
    X = np.column_stack([own - own.mean(0), np.ones(len(own))])
    beta, *_ = np.linalg.lstsq(X, x, rcond=None)
    return x - X @ beta


def standardize(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-12)


def cca(a, b, eps=1e-6):
    """Canonical correlations and the weights of each side, via SVD."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    n = len(a)
    Wa, Wb = inv_sqrt(a.T @ a / n, eps), inv_sqrt(b.T @ b / n, eps)
    U, s, Vt = np.linalg.svd(Wa @ (a.T @ b / n) @ Wb, full_matrices=False)
    return s, Wa @ U, Wb @ Vt.T


def unit(W):
    return W / np.linalg.norm(W, axis=0)


def r2(t, X):
    """R^2 of t linearly predicted by X (with intercept), used for the composition only."""
    Xc = np.column_stack([X - X.mean(0), np.ones(len(X))])
    beta, *_ = np.linalg.lstsq(Xc, t, rcond=None)
    return float(1 - np.var(t - Xc @ beta) / np.var(t))


def group_bases(d_raw, n_mates, p, blocks=('adversary',)):
    """Orthonormal bases (raw coordinates) of the centroid / spread subspaces and each later block."""
    mate_dims = n_mates * p
    cen = np.zeros((d_raw, p))
    for j in range(n_mates):
        cen[j * p:(j + 1) * p] = np.eye(p)
    cen /= np.sqrt(n_mates)
    proj = np.eye(d_raw)[:, :mate_dims] - cen @ (cen.T @ np.eye(d_raw)[:, :mate_dims])
    U, s, _ = np.linalg.svd(proj, full_matrices=False)
    out = {'centroid': cen, 'difference': U[:, s > 1e-8]}
    for b, name in enumerate(blocks):
        out[name] = np.eye(d_raw)[:, mate_dims + b * p:mate_dims + (b + 1) * p]
    return out


def physical_bases(z):
    blocks = ['adversary'] + [name for name, _ in SPEC['extras']]
    return group_bases(z.shape[1], len(SPEC['agents']) - 1, SPEC['p'], blocks)


def composition(F, s):
    """For unit raw functionals F (columns): R^2 from each physical group, and the geometric share
    of each group's subspace (the row labels in the figures)."""
    n_mates, p = len(SPEC['agents']) - 1, SPEC['p']
    z = s['z']
    zc = z - z.mean(0)
    mates = z[:, :n_mates * p].reshape(len(z), n_mates, p)
    cen = mates.mean(1)
    feats = {'adversary': s['adv'], 'centroid': cen, 'difference': (mates - cen[:, None]).reshape(len(z), -1)}
    feats.update({name: s[name] for name, _ in SPEC['extras']})
    bases = physical_bases(z)
    out = {'r2': {g: [] for g in feats}, 'share': {g: [] for g in bases}}
    for f in F.T:
        t = zc @ f
        for g, X in feats.items():
            out['r2'][g].append(r2(t, X))
        for g, Bg in bases.items():
            out['share'][g].append(float(np.sum((Bg.T @ f) ** 2)))
    return out


# ---------------------------------------------------------------- analysis

def task_cca(s):
    """CCA of the whitened true state with the controller targets, own state regressed out of both.
    Returns the squared canonical correlations and the unit task functionals in raw coordinates."""
    zc = s['z'] - s['z'].mean(0)
    W = inv_sqrt(np.cov(zc, rowvar=False))
    zt_res = residualize(zc @ W, s['own'])
    st, Tw, _ = cca(zt_res, standardize(residualize(s['y'], s['own'])))
    return st ** 2, unit(W @ unit(Tw))


def axes(sets, min_eig, r_ref=None):
    """Task functionals from the true model (fixed for every method)."""
    eig, F_all = task_cca(sets['fully_obs'])
    if r_ref is None:
        r_ref = int((eig > min_eig).sum())
    return r_ref, F_all[:, :r_ref], eig, F_all


def r2_along(err, truth):
    """Multivariate R^2 = 1 - sum of squared errors / total variance (columns already projected)."""
    return float(1 - np.sum(err ** 2) / np.sum((truth - truth.mean(0)) ** 2))


def evaluate(sets, min_eig, r_ref=None):
    """Raw R^2 of each belief (its own estimate, exactly what the policy receives) on the fixed axes."""
    r_ref, F_task, eig, F_all = axes(sets, min_eig, r_ref)
    groups = physical_bases(sets['fully_obs']['z'])
    out = {'r_ref': r_ref, 'task_eig': eig[:r_ref], 'task_eig_all': eig}
    for m in METHODS:
        z, zh = sets[m]['z'], sets[m]['zh']
        err = zh - z
        task = np.array([r2_along(err @ F_task[:, [k]], z @ F_task[:, [k]]) for k in range(r_ref)])
        out[m] = {'raw': dict(
            task=task,
            task_all=np.array([r2_along(err @ F_all[:, [k]], z @ F_all[:, [k]]) for k in range(F_all.shape[1])]),
            coverage=float(task.mean()),  # mean over the fixed controller task directions
            physical={g: r2_along(err @ B, z @ B) for g, B in groups.items()},
        )}
    return out


def flatten(x, prefix=''):
    """Flatten numeric leaves to {path: array}."""
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flatten(v, f'{prefix}{k}/'))
    elif x is not None:
        out[prefix.rstrip('/')] = np.asarray(x, dtype=float)
    return out


def to_json(x):
    if isinstance(x, dict):
        return {k: to_json(v) for k, v in x.items()}
    if isinstance(x, (np.ndarray, list, tuple)):
        return np.asarray(x, dtype=float).round(5).tolist()
    if isinstance(x, (np.floating, float)):
        return round(float(x), 5)
    return x


# ---------------------------------------------------------------- figure data

def projections(sets, F, n_keep, seed):
    """Truth and the raw belief estimate projected on the fixed task directions."""
    rng = np.random.default_rng(seed)
    out = {'functionals': F}
    for m in METHODS:
        z, zh = sets[m]['z'], sets[m]['zh']
        idx = rng.choice(len(z), size=min(n_keep, len(z)), replace=False)
        out[f'{m}/actual'] = z[idx] @ F
        out[f'{m}/belief'] = zh[idx] @ F
    return out


def centroid_samples(sets, n_keep, seed):
    """True and raw belief teammate centroid (x, y; relative to the agent) for the centroid map."""
    rng = np.random.default_rng(seed)
    n_mates, p = len(SPEC['agents']) - 1, SPEC['p']
    out = {}
    for m in METHODS:
        z, zh = sets[m]['z'], sets[m]['zh']
        idx = rng.choice(len(z), size=min(n_keep, len(z)), replace=False)
        cen = lambda x: x[idx, :n_mates * p].reshape(len(idx), n_mates, p).mean(1)[:, :2]
        out[f'{m}/true'] = cen(z)
        out[f'{m}/raw'] = cen(zh)
        # total variance of the full centroid over all samples: the R^2 denominator
        full = z[:, :n_mates * p].reshape(len(z), n_mates, p).mean(1)[:, :2]
        out[f'{m}/total_var'] = np.array(np.sum(np.var(full, 0)))
    return out


def episode_r2(sets, F):
    """Per-episode raw R^2 along each task functional: 1 - mean episode squared error / variance of the
    truth over ALL of that method's data (so episodes are comparable and average to the overall R^2)."""
    out = {}
    for m in METHODS:
        z, zh, ep = sets[m]['z'], sets[m]['zh'], sets[m]['ep']
        t, e = z @ F, (zh - z) @ F
        var = t.var(0)
        rows = episode_rows(ep)
        out[m] = np.stack([1 - (e[r] ** 2).mean(0) / var for r in rows])  # (n_episodes, n_directions)
        out[f'{m}/n_steps'] = np.array([len(r) for r in rows])
    return out


def save_figure_data(sets, est, min_eig, out_dir, seed):
    _, F_task, _, _ = axes(sets, min_eig)
    np.savez(os.path.join(out_dir, 'episode_r2.npz'), **episode_r2(sets, F_task))
    proj = projections(sets, F_task, 80000, seed)
    proj['task_eig'] = est['task_eig']
    for g, v in composition(F_task, sets['fully_obs'])['share'].items():
        proj[f'share/{g}'] = np.array(v)
    np.savez(os.path.join(out_dir, 'projections.npz'), **proj)
    np.savez(os.path.join(out_dir, 'centroid.npz'), **centroid_samples(sets, 200000, seed))


# ---------------------------------------------------------------- actions and success

def episode_actions(path):
    """Per episode: steps where the action taken equals the policy's action on the true observation,
    agent-steps compared, success flag and length (full episode)."""
    d = np.load(path, allow_pickle=True)
    steps = sorted(int(k) for k in d.files if k.isdigit())
    agree = total = 0
    for k in steps:
        c = d[str(k)][4]['__common__']
        taken, true = c.get('policy_actions', {}), c.get('policy_actions_true_obs', {})
        for a, act in taken.items():
            if a in true:
                total += 1
                agree += int(np.asarray(act).item() == np.asarray(true[a]).item())
    return agree, total, SPEC['success'](d[str(steps[-1])]), len(steps)


def action_metrics(dirs, n_boot, seed, workers):
    """Action agreement and success rate per method, with an episode bootstrap."""
    rng = np.random.default_rng(seed)
    est, boot = {}, {}
    for m, dpath in dirs.items():
        files = sorted(glob.glob(os.path.join(dpath, '*.npz')))
        with Pool(workers) as pool:
            r = np.array(pool.map(episode_actions, files), dtype=float)
        if r[:, 1].sum() == 0:
            print(f'{m}: no action logging in {dpath}', flush=True)
            continue
        def stats(i):
            return np.array([r[i, 0].sum() / r[i, 1].sum(), r[i, 2].mean(), r[i, 3].mean()])
        est[m] = stats(np.arange(len(r)))
        boot[m] = np.stack([stats(rng.integers(len(r), size=len(r))) for _ in range(n_boot)])
    names = ['action_agreement', 'success_rate', 'mean_episode_length']
    return ({m: dict(zip(names, v.tolist())) for m, v in est.items()},
            {f'actions/{m}/{n}': boot[m][:, i] for m in boot for i, n in enumerate(names)})


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', choices=list(ENVS), required=True)
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--min-eig', type=float, default=0.1)
    p.add_argument('--n-boot', type=int, default=100)
    p.add_argument('--workers', type=int, default=48)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--limit-episodes', type=int, default=None, help='for testing only')
    p.add_argument('--out-dir', default=None, help='default results/<env>/cca_final')
    p.add_argument('--data-dir', default=None,
                   help='checkpoint folder holding results/, results_allocentric/, results_pf/ '
                        '(default: the final checkpoint in ENVS)')
    p.add_argument('--figdata-only', action='store_true',
                   help='only rewrite episode_r2.npz, projections.npz and centroid.npz '
                        '(no bootstrap, keeps the existing summary)')
    args = p.parse_args()

    SPEC.update(ENVS[args.env])
    SPEC['cfg'] = (yaml.safe_load(open(os.path.join(REPO, SPEC['config'])))['env']['controller_kwargs']
                   if SPEC['config'] else None)
    out_dir = args.out_dir or os.path.join('results', args.env, 'cca_final')
    os.makedirs(out_dir, exist_ok=True)
    data_dir = args.data_dir or SPEC['checkpoint']
    dirs = {m: os.path.join(data_dir, CHECKPOINT_FOLDERS[m]) for m in METHODS}

    sets = {}
    for m in METHODS:
        sets[m] = load_method(dirs[m], args.workers, args.max_steps, args.limit_episodes)
        print(f'loaded {m}: {len(sets[m]["ep"])} agent-steps', flush=True)

    est = evaluate(sets, args.min_eig)
    save_figure_data(sets, est, args.min_eig, out_dir, args.seed)
    if args.figdata_only:
        print('FIGURE DATA DONE', flush=True)
        return

    rng = np.random.default_rng(args.seed)
    rows = {m: episode_rows(sets[m]['ep']) for m in METHODS}
    boots = []
    for b in range(args.n_boot):
        bs = {}
        for m in METHODS:
            idx = np.concatenate([rows[m][i] for i in rng.integers(len(rows[m]), size=len(rows[m]))])
            bs[m] = {k: v[idx] for k, v in sets[m].items()}
        boots.append(flatten(evaluate(bs, args.min_eig, r_ref=est['r_ref'])))
        if b % 20 == 0:
            print(f'bootstrap {b}/{args.n_boot}', flush=True)
    ci = {k: np.percentile(np.stack([bb[k] for bb in boots]), [2.5, 97.5], axis=0).tolist() for k in boots[0]}
    np.savez(os.path.join(out_dir, 'bootstrap.npz'), **{k: np.stack([bb[k] for bb in boots]) for k in boots[0]})

    _, F_task, _, F_all = axes(sets, args.min_eig)
    actions, actions_boot = action_metrics(dirs, args.n_boot, args.seed, args.workers)
    print('actions:', json.dumps(actions), flush=True)
    np.savez(os.path.join(out_dir, 'bootstrap_actions.npz'), **actions_boot)

    summary = {
        'settings': dict(vars(args), out_dir=out_dir, data_dirs=dirs),
        'actions': actions,
        'meta': dict(env=args.env, adversary=SPEC['adversary'], title=title(),
                     n_teammates=len(SPEC['agents']) - 1),
        'task_composition': to_json(composition(F_task, sets['fully_obs'])),
        'task_composition_all': to_json(composition(F_all, sets['fully_obs'])),
        'estimate': to_json(est),
        'ci95': ci,
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'saved {out_dir}/summary.json', flush=True)
    print('ANALYSIS DONE', flush=True)


if __name__ == '__main__':
    main()
