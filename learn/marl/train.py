import os
import json
import torch
import shutil
import pickle
import copy
import multiprocessing as mp
import numpy as np
from torch import nn
from typing import Optional
from gymnasium import spaces
from concurrent.futures import ProcessPoolExecutor
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

#from logger import getlogger
from util.util import mkdir, load_config
from envs.marl.make_env import make_marl_env
from controllers.marl_slot_controller import compute_slot_actions
from controllers.drone_slot_controller import compute_drone_slot_actions
from learn.marl.callbacks import LogRawEpisodeReturn


#logger = getlogger(__name__)

def _load_checkpoint_iteration(checkpoint_dir: str) -> int | None:
    state_path = os.path.join(checkpoint_dir, 'algorithm_state.pkl')
    if not os.path.exists(state_path):
        return None

    try:
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
    except Exception:
        return None

    training_iteration = state.get('training_iteration')
    if training_iteration is None:
        return None

    return int(training_iteration)

def _find_latest_checkpoint(logdir: str) -> tuple[str | None, int]:
    latest_dir = None
    latest_iter = -1

    logdir = os.path.abspath(logdir)

    if not os.path.exists(logdir):
        return None, latest_iter

    for entry in os.scandir(logdir):
        if not entry.is_dir():
            continue

        training_iteration = _load_checkpoint_iteration(entry.path)
        if training_iteration is None:
            continue

        if training_iteration > latest_iter:
            latest_iter = training_iteration
            latest_dir = entry.path

    return latest_dir, latest_iter

def _save_checkpoint(algo, checkpoint_dir: str):
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    algo.save(checkpoint_dir=checkpoint_dir)

def _safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default

def _best_checkpoint_metrics(result: dict) -> tuple[tuple[float, float, float], dict[str, float]]:
    env_stats = result.get('env_runners') or result.get('sampler_results') or {}
    custom_metrics = env_stats.get('custom_metrics') or result.get('custom_metrics') or {}

    summary = {
        'training_iteration': _safe_float(result.get('training_iteration'), 0.0),
        'success_mean': _safe_float(custom_metrics.get('success_mean'), 0.0),
        'hold_steps_max_mean': _safe_float(custom_metrics.get('hold_steps_max_mean'), 0.0),
        'episode_return_mean': _safe_float(
            env_stats.get('episode_return_mean', result.get('episode_reward_mean')),
            -1.0e30,
        ),
        'coverage_score_mean_mean': _safe_float(custom_metrics.get('coverage_score_mean_mean'), 0.0),
        'close_fraction_mean_mean': _safe_float(custom_metrics.get('close_fraction_mean_mean'), 0.0),
        'ring_score_mean_mean': _safe_float(custom_metrics.get('ring_score_mean_mean'), 0.0),
        'slot_score_mean_mean': _safe_float(custom_metrics.get('slot_score_mean_mean'), 0.0),
        'control_score_mean_mean': _safe_float(custom_metrics.get('control_score_mean_mean'), 0.0),
        'goal_center_score_mean_mean': _safe_float(custom_metrics.get('goal_center_score_mean_mean'), 0.0),
        'goal_lock_score_mean_mean': _safe_float(custom_metrics.get('goal_lock_score_mean_mean'), 0.0),
        'target_goal_dist_final_mean': _safe_float(custom_metrics.get('target_goal_dist_final_mean'), 0.0),
        'target_goal_dist_min_mean': _safe_float(custom_metrics.get('target_goal_dist_min_mean'), 0.0),
        'oob_mean': _safe_float(custom_metrics.get('oob_mean'), 1.0),
    }

    # Success is sparse here, so rank checkpoints lexicographically instead of
    # letting the dense shaped return hide a rare actual hold.
    rank = (
        summary['success_mean'],
        summary['hold_steps_max_mean'],
        summary['episode_return_mean'],
    )
    return rank, summary

def _build_algorithm(algo_config, logger_creator):
    # Ray has bounced between build() and build_algo(); the installed lab
    # version should decide which API we use.
    if hasattr(algo_config, 'build_algo'):
        return algo_config.build_algo(logger_creator=logger_creator)
    return algo_config.build(logger_creator=logger_creator)

def _checkpoint_path_candidates(path: str) -> list[str]:
    candidates = [path]
    if path.startswith('logs' + os.sep):
        candidates.append(os.path.join('lab-logs', path[len('logs' + os.sep):]))
    if not os.path.isabs(path):
        candidates.extend(os.path.abspath(candidate) for candidate in list(candidates))
    return list(dict.fromkeys(candidates))

def _resolve_checkpoint_path(path: str) -> str:
    candidates = _checkpoint_path_candidates(path)
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        'Initial checkpoint not found. Tried:\n  ' + '\n  '.join(candidates)
    )

def _apply_alg_env_overrides(cfg: dict):
    overrides = {
        'TELL_NENVS': ('nenvs', int),
        'TELL_CPU_ENVS': ('cpu_envs', int),
        'TELL_NUM_CPUS_PER_ENV_RUNNER': ('num_cpus_per_env_runner', int),
        'TELL_NUM_CPUS_FOR_MAIN_PROCESS': ('num_cpus_for_main_process', int),
        'TELL_BATCH': ('batch', int),
        'TELL_MINIBATCH_SIZE': ('minibatch_size', int),
        'TELL_NUM_EPOCHS': ('num_epochs', int),
        'TELL_NUM_GPUS': ('num_gpus', int),
        'TELL_SAMPLE_TIMEOUT_S': ('sample_timeout_s', float),
    }
    applied = {}
    for env_name, (cfg_key, caster) in overrides.items():
        value = os.environ.get(env_name)
        if value is None:
            continue
        cfg['alg'][cfg_key] = caster(value)
        applied[cfg_key] = cfg['alg'][cfg_key]
    pretrain_workers = os.environ.get('TELL_PRETRAIN_WORKERS')
    if pretrain_workers is not None:
        cfg['alg'].setdefault('pretrain', {})['workers'] = int(pretrain_workers)
        applied['pretrain.workers'] = cfg['alg']['pretrain']['workers']
    rollout_fragment_length = os.environ.get('TELL_ROLLOUT_FRAGMENT_LENGTH')
    if rollout_fragment_length is not None:
        cfg['alg']['rollout_fragment_length'] = (
            rollout_fragment_length
            if rollout_fragment_length == 'auto'
            else int(rollout_fragment_length)
        )
        applied['rollout_fragment_length'] = cfg['alg']['rollout_fragment_length']
    if applied:
        print(f'Applied runtime training overrides: {applied}')

def _pretrain_algorithm_config(cfg: dict) -> dict:
    pretrain_cfg = copy.deepcopy(cfg)
    pretrain_settings = pretrain_cfg['alg'].get('pretrain', {})
    pretrain_cfg['alg']['nenvs'] = int(pretrain_settings.get('nenvs', 0))
    pretrain_cfg['alg']['cpu_envs'] = int(pretrain_settings.get('cpu_envs', 1))
    return pretrain_cfg

def _episodes_done(terminations: dict, truncations: dict) -> bool:
    return all(bool(v) for v in terminations.values()) or all(bool(v) for v in truncations.values())

def _summarize_eval_rows(rows: list[dict]) -> dict[str, float]:
    metrics = {key: np.array([row[key] for row in rows], dtype=float) for key in rows[0]}
    return {
        'success_rate': float(metrics['success'].mean()),
        'oob_rate': float(metrics['oob'].mean()),
        'avg_max_hold': float(metrics['max_hold'].mean()),
        'avg_best_dist': float(metrics['best_dist'].mean()),
        'avg_final_dist': float(metrics['final_dist'].mean()),
        'avg_steps': float(metrics['steps'].mean()),
    }

def _format_policy_action(action, action_space):
    if isinstance(action, tuple):
        action = action[0]
    if isinstance(action_space, spaces.Discrete):
        return int(action)
    return np.clip(np.asarray(action, dtype=np.float32), action_space.low, action_space.high)

def _evaluate_shared_policy(algo, cfg: dict, runs: int = 20) -> dict[str, float]:
    policy_id = cfg['policy_list'][0]
    action_space = algo.get_policy(policy_id).action_space
    rows = []

    for seed_offset in range(runs):
        env = make_marl_env(cfg, seed=int(cfg['seed']) + seed_offset, wrap=None)
        try:
            obs, _ = env.reset()

            best_dist = float('inf')
            max_hold = 0
            success = False
            oob = False
            steps = 0

            for t in range(cfg['env']['max_episode_length']):
                action_dict = {}
                for agent in cfg['env']['learned_agent_list']:
                    action = algo.compute_single_action(obs[agent], policy_id=policy_id, explore=False)
                    action_dict[agent] = _format_policy_action(action, action_space)

                obs, _, terminations, truncations, infos = env.step(action_dict)
                info = infos[cfg['env']['learned_agent_list'][0]]

                best_dist = min(best_dist, float(info['target_goal_dist']))
                max_hold = max(max_hold, int(info.get('hold_steps', 0)))
                success = success or bool(info.get('success', False))
                oob = oob or bool(info.get('oob', False))
                steps = t + 1

                if _episodes_done(terminations, truncations):
                    break

            final_info = infos[cfg['env']['learned_agent_list'][0]]
            rows.append({
                'success': int(success),
                'oob': int(oob),
                'max_hold': max_hold,
                'best_dist': best_dist,
                'final_dist': float(final_info['target_goal_dist']),
                'steps': steps,
            })
        finally:
            env.close()

    return _summarize_eval_rows(rows)

def _collect_slot_controller_episode(cfg: dict, episode_idx: int):
    obs_buf = []
    act_buf = []
    ret_buf = []
    gamma = float(cfg['alg']['gamma'])
    learned_agents = cfg['env']['learned_agent_list']

    env = make_marl_env(
        cfg,
        seed=int(cfg['seed']) + episode_idx,
        wrap=None,
    )
    try:
        obs, _ = env.reset()
        trajectory = []

        for _ in range(cfg['env']['max_episode_length']):
            if 'drones' in cfg['env']['scenario']:
                expert_actions = compute_drone_slot_actions(env)
            else:
                expert_actions = compute_slot_actions(env.env.unwrapped.world)
            predator_obs = {
                agent: np.asarray(obs[agent], dtype=np.float32).copy()
                for agent in learned_agents
            }
            action_dict = {**expert_actions}
            obs, rewards, terminations, truncations, _ = env.step(action_dict)
            team_reward = float(rewards[learned_agents[0]])
            trajectory.append((predator_obs, expert_actions, team_reward))

            if _episodes_done(terminations, truncations):
                break
    finally:
        env.close()

    discounted_returns = []
    ret = 0.0
    for _, _, reward in reversed(trajectory):
        ret = reward + gamma * ret
        discounted_returns.append(ret)
    discounted_returns.reverse()

    for (predator_obs, expert_actions, _), discounted_return in zip(trajectory, discounted_returns):
        for agent in learned_agents:
            obs_buf.append(predator_obs[agent])
            act_buf.append(np.asarray(expert_actions[agent]).copy())
            ret_buf.append(float(discounted_return))

    return obs_buf, act_buf, ret_buf

def _collect_slot_controller_chunk(args):
    cfg, episode_indices = args
    obs_buf = []
    act_buf = []
    ret_buf = []

    for episode_idx in episode_indices:
        episode_obs, episode_act, episode_ret = _collect_slot_controller_episode(cfg, episode_idx)
        obs_buf.extend(episode_obs)
        act_buf.extend(episode_act)
        ret_buf.extend(episode_ret)

    return _pack_controller_dataset(obs_buf, act_buf, ret_buf)

def _pack_controller_dataset(obs_buf, act_buf, ret_buf):
    act_arr = np.asarray(act_buf)
    if act_arr.ndim == 1:
        act_arr = act_arr.astype(np.int64)
    else:
        act_arr = act_arr.astype(np.float32)

    return (
        np.stack(obs_buf),
        act_arr,
        np.asarray(ret_buf, dtype=np.float32),
    )

def _episode_chunks(episodes: int, workers: int) -> list[list[int]]:
    workers = max(1, min(workers, episodes))
    return [
        list(range(worker_idx, episodes, workers))
        for worker_idx in range(workers)
    ]

def _collect_slot_controller_dataset(cfg: dict, episodes: int, workers: int = 1):
    if workers <= 1:
        obs_buf = []
        act_buf = []
        ret_buf = []
        for episode_idx in range(episodes):
            episode_obs, episode_act, episode_ret = _collect_slot_controller_episode(cfg, episode_idx)
            obs_buf.extend(episode_obs)
            act_buf.extend(episode_act)
            ret_buf.extend(episode_ret)
        return _pack_controller_dataset(obs_buf, act_buf, ret_buf)

    obs_chunks = []
    act_chunks = []
    ret_chunks = []
    chunks = _episode_chunks(episodes, workers)
    context = mp.get_context('spawn')
    # Use process-level parallelism for pretraining data, but keep the cap
    # below the PyBullet connection limit that bit us in earlier runs.
    with ProcessPoolExecutor(max_workers=len(chunks), mp_context=context) as executor:
        for chunk_obs, chunk_act, chunk_ret in executor.map(
            _collect_slot_controller_chunk,
            [(cfg, chunk) for chunk in chunks],
        ):
            obs_chunks.append(chunk_obs)
            act_chunks.append(chunk_act)
            ret_chunks.append(chunk_ret)

    return (
        np.concatenate(obs_chunks, axis=0),
        np.concatenate(act_chunks, axis=0),
        np.concatenate(ret_chunks, axis=0),
    )

def _maybe_pretrain_policy(algo, cfg: dict, logdir: str) -> bool:
    pretrain_cfg = cfg['alg'].get('pretrain', {})

    if not pretrain_cfg or not pretrain_cfg.get('enabled', False):
        return False

    if 'ppo' not in cfg['alg']['type']:
        print('Skipping expert pretrain because it is only implemented for PPO.')
        return False

    episodes = int(pretrain_cfg.get('episodes', 300))
    epochs = int(pretrain_cfg.get('epochs', 20))
    batch_size = int(pretrain_cfg.get('batch_size', 1024))
    learning_rate = float(pretrain_cfg.get('lr', 1e-3))
    value_loss_coef = float(pretrain_cfg.get('value_loss_coef', 0.05))
    grad_clip = float(pretrain_cfg.get('grad_clip', 1.0))
    eval_runs = int(pretrain_cfg.get('eval_runs', 20))
    workers = int(pretrain_cfg.get('workers', 1))
    policy_id = cfg['policy_list'][0]

    print(f'[abm] Collecting {episodes} expert episodes from the slot controller with {workers} workers...')
    obs_arr, act_arr, ret_arr = _collect_slot_controller_dataset(
        cfg,
        episodes=episodes,
        workers=workers,
    )
    action_space = algo.get_policy(policy_id).action_space
    if isinstance(action_space, spaces.Discrete):
        action_summary = {
            'action_counts': {
                action: int((act_arr == action).sum())
                for action in range(int(action_space.n))
            }
        }
    else:
        action_summary = {
            'action_mean': act_arr.mean(axis=0).round(3).tolist(),
            'action_std': act_arr.std(axis=0).round(3).tolist(),
        }
    print(
        'Expert dataset:',
        {
            'samples': int(obs_arr.shape[0]),
            'obs_dim': int(obs_arr.shape[1]),
            **action_summary,
        },
    )

    policy = algo.get_policy(policy_id)
    model = policy.model
    device = next(model.parameters()).device
    model.train()

    if isinstance(action_space, spaces.Discrete):
        class_counts = np.bincount(act_arr, minlength=int(action_space.n))
        class_weights = class_counts.sum() / np.maximum(class_counts, 1)
        class_weights = class_weights / class_weights.mean()
        policy_loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
        )
        act_tensor = torch.tensor(act_arr, dtype=torch.long)
    else:
        policy_loss_fn = nn.MSELoss()
        act_tensor = torch.tensor(act_arr, dtype=torch.float32)
    value_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(
        torch.tensor(obs_arr, dtype=torch.float32),
        act_tensor,
        torch.tensor(ret_arr, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_ce = 0.0
        total_value = 0.0
        total_correct = 0.0
        total_action_l2 = 0.0
        total_samples = 0

        for obs_batch, act_batch, ret_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            ret_batch = ret_batch.to(device)

            logits, _ = model({'obs': obs_batch}, [], None)
            values = model.value_function()

            # This is straight behavior cloning on the policy head, with a light
            # value-head fit so PPO does not start from a completely useless critic.
            if isinstance(action_space, spaces.Discrete):
                policy_loss = policy_loss_fn(logits, act_batch)
                total_correct += float((logits.argmax(dim=-1) == act_batch).float().sum().item())
            else:
                action_dim = int(np.prod(action_space.shape))
                pred_action = logits[:, :action_dim]
                policy_loss = policy_loss_fn(pred_action, act_batch)
                total_action_l2 += float(
                    torch.linalg.norm(pred_action - act_batch, dim=-1).sum().item()
                )
            critic_loss = value_loss(values, ret_batch)
            loss = policy_loss + (value_loss_coef * critic_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_ce += float(policy_loss.item()) * len(obs_batch)
            total_value += float(critic_loss.item()) * len(obs_batch)
            total_samples += len(obs_batch)

        epoch_metrics = {
            'policy_loss': total_ce / total_samples,
            'value_mse': total_value / total_samples,
        }
        if isinstance(action_space, spaces.Discrete):
            epoch_metrics['accuracy'] = total_correct / total_samples
        else:
            epoch_metrics['action_l2'] = total_action_l2 / total_samples
        print(f'Pretrain epoch {epoch + 1}/{epochs}:', epoch_metrics)

    summary = _evaluate_shared_policy(algo, cfg, runs=eval_runs)
    print(f'Pretrain evaluation over {eval_runs} runs: {summary}')

    pretrain_dir = os.path.join(logdir, 'checkpoint0_pretrain')
    _save_checkpoint(algo, pretrain_dir)
    _save_checkpoint(algo, os.path.join(logdir, 'checkpoint_latest'))
    algo.restore(pretrain_dir)

    return True

def train(config_path: str):

    cfg = load_config(config_path)
    _apply_alg_env_overrides(cfg)
    logdir = os.path.abspath(cfg['logdir'])
    cfg['logdir'] = logdir
    if not os.path.exists(logdir):
        print('Save directory not found, creating path ...')
        #logger.info("Save directory not found, creating path ...")
        mkdir(logdir)
    else:
        print('Save directory found ...')
        #logger.info("Save directory found ...")
    print("current directory:", logdir)

    resume_dir, resume_iter = _find_latest_checkpoint(logdir)
    initial_checkpoint = None
    if resume_dir is None:
        initial_checkpoint = cfg['alg'].get('initial_checkpoint')
        if initial_checkpoint:
            initial_checkpoint = _resolve_checkpoint_path(initial_checkpoint)
            cfg['alg']['initial_checkpoint'] = initial_checkpoint

    #save copy of config file
    OmegaConf.save(config=cfg, f=logdir+'/config.yaml')

    def logger_creator(config):
        try:
            from ray.tune.logger import UnifiedLogger
            return UnifiedLogger(config, logdir, loggers=None)
        except:
            print('Cannot import unified logger, check ray version...')

    start_iter = 0
    if resume_dir is not None:
        algo_config = make_ray_config(cfg)
        algo_build = _build_algorithm(algo_config, logger_creator)
        print(f'Resuming from checkpoint: {resume_dir}')
        algo_build.restore(resume_dir)
        start_iter = max(resume_iter, 0)
        print(f'Restored training iteration: {start_iter}')
    else:
        if initial_checkpoint:
            algo_config = make_ray_config(cfg)
            algo_build = _build_algorithm(algo_config, logger_creator)
            print(f'Initializing from checkpoint: {initial_checkpoint}')
            algo_build.restore(initial_checkpoint)
        else:
            pretrain_checkpoint = None
            if cfg['alg'].get('pretrain', {}).get('enabled', False):
                pretrain_cfg = _pretrain_algorithm_config(cfg)
                print(
                    'Building small pretrain policy before launching full Ray workers:',
                    {
                        'nenvs': pretrain_cfg['alg']['nenvs'],
                        'cpu_envs': pretrain_cfg['alg']['cpu_envs'],
                    },
                )
                pretrain_algo = _build_algorithm(make_ray_config(pretrain_cfg), logger_creator)
                try:
                    if _maybe_pretrain_policy(pretrain_algo, cfg, logdir):
                        pretrain_checkpoint = os.path.join(logdir, 'checkpoint0_pretrain')
                finally:
                    pretrain_algo.stop()

            algo_config = make_ray_config(cfg)
            algo_build = _build_algorithm(algo_config, logger_creator)
            if pretrain_checkpoint is not None:
                print(f'Initializing full training from pretrain checkpoint: {pretrain_checkpoint}')
                algo_build.restore(pretrain_checkpoint)

    # Config owns the PPO budget; the lab runs are intentionally short enough
    # that we can compare checkpoints instead of blindly trusting the final one.
    total_iters = int(cfg['alg']['timesteps'])
    checkpoint_freq = int(cfg['alg'].get('checkpoint_freq', 500))
    best_rank: tuple[float, float, float] | None = None
    best_metrics_path = os.path.join(logdir, 'checkpoint_best_metrics.json')
    if os.path.exists(best_metrics_path):
        with open(best_metrics_path, 'r') as f:
            prior_best = json.load(f)
        best_rank = (
            _safe_float(prior_best.get('success_mean'), 0.0),
            _safe_float(prior_best.get('hold_steps_max_mean'), 0.0),
            _safe_float(prior_best.get('episode_return_mean'), -1.0e30),
        )

    interrupted = False
    try:
        try:
            for i in range(start_iter, total_iters):
                result = algo_build.train()
                current_iter = i + 1

                rank, best_metrics = _best_checkpoint_metrics(result)
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_dir = os.path.join(logdir, 'checkpoint_best')
                    _save_checkpoint(algo_build, best_dir)
                    with open(os.path.join(logdir, 'checkpoint_best_metrics.json'), 'w') as f:
                        json.dump(best_metrics, f, indent=2, sort_keys=True)
                    print(f'Updated best checkpoint at iteration {current_iter}: {best_metrics}')

                if current_iter % checkpoint_freq == 0:
                    save_dir = os.path.join(logdir, 'checkpoint' + str(current_iter))
                    _save_checkpoint(algo_build, save_dir)
                    _save_checkpoint(algo_build, os.path.join(logdir, 'checkpoint_latest'))
                    print(pretty_print(result))
        except KeyboardInterrupt:
            interrupted = True
            interrupt_dir = os.path.join(logdir, 'checkpoint_latest')
            print(f'\nTraining interrupted. Saving latest checkpoint to: {interrupt_dir}')
            _save_checkpoint(algo_build, interrupt_dir)

        if not interrupted:
            save_dir = os.path.join(logdir, 'final')
            _save_checkpoint(algo_build, save_dir)
            _save_checkpoint(algo_build, os.path.join(logdir, 'checkpoint_latest'))
    finally:
        algo_build.stop()


def make_ray_config(
    cfg: dict,
    wrapper: Optional[str] = 'rllib',
):
    '''
    takes loaded config and returns ray config file for tarining rl

    input
    -----
    config:dict
        loaded config dictionary

    output
    ------
    ray_config class
    '''
    #make env function 
    def env_maker(config):

        seed = cfg['seed']

        if config is not None:
            seed += ((1000*config.worker_index + config.vector_index))   

        env = make_marl_env(cfg,seed=int(seed),wrap=wrapper)

        return env
        
    class policyTrainingSchedule():
        '''
        records how many batches have been used 
        '''
        def __init__(self,workers=1):
            self.workers=workers
            self.max_samples = 0

        def policy_training_schedule(self, pid, batch):
            '''
            determines when each policy should be trained
            '''
            return True
            
    policy_tracker = policyTrainingSchedule(workers=cfg['alg']['nenvs'])
    
    env_name = cfg['env']['scenario']
    register_env(env_name, env_maker) #register make env function
    #test_env for getting obs/action space
    test_env = env_maker(None)
    policy_list = cfg['policy_list']
    policy_mapping_fn = marl_policy_mapping_fn
    policy_training_fn = policy_tracker.policy_training_schedule

    batch = None

    training_kwargs = {}

    if 'sac' in cfg['alg']['type']:

        algo = SACConfig() 
        policy_model_dict = {
            'post_fcnet_hiddens': cfg['alg']['pi'],
        }
        q_model_dict = {
            'post_fcnet_hiddens': cfg['alg']['vf'],
        }
        batch = cfg['alg']['batch']*len(cfg['policy_list'])

        policy_info = {}

        for label in policy_list:

            policy_info[label] = PolicySpec(
                            None, #policy_class
                            test_env.get_observation_space(label), #observation_space
                            test_env.get_action_space(label), #action_space
                            {'lr':cfg['alg']['lr'],
                                'policy_model_config':policy_model_dict,
                                'q_model_config':q_model_dict,
                            } #config (gamma,lr,etc.)
                        )
        training_kwargs = {
            'gamma': cfg['alg']['gamma'],
            'train_batch_size': batch,
            'training_intensity': None,
            'grad_clip_by': 'norm',
            'replay_buffer_config': {
                'type': 'MultiAgentReplayBuffer',
                'capacity': 1000000,
                'replay_sequence_length': 1,
            },
        }
    elif 'ppo' in cfg['alg']['type']:
        algo = PPOConfig()
        model_dict = {
            'fcnet_hiddens': cfg['alg']['pi'],
        }
        batch = cfg['alg']['batch']

        policy_info = {}
        for label in policy_list:
            policy_info[label] = PolicySpec(
                None,
                test_env.get_observation_space(label),
                test_env.get_action_space(label),
                {},
            )

        training_kwargs = {
            'gamma': cfg['alg']['gamma'],
            'lr': cfg['alg']['lr'],
            'train_batch_size': batch,
            'model': model_dict,
            'lambda_': cfg['alg'].get('lambda', 0.95),
            'clip_param': cfg['alg'].get('clip_param', 0.2),
            'entropy_coeff': cfg['alg'].get('entropy_coeff', 0.01),
            'vf_clip_param': cfg['alg'].get('vf_clip_param', 10.0),
            'grad_clip': cfg['alg'].get('grad_clip', 0.5),
        }
    else:
        raise ValueError(f"Unsupported MARL algorithm type: {cfg['alg']['type']}")

    #initialize RL training algorithm using rllib's multi_agent settings
    env_runner_kwargs = {
        'num_env_runners': cfg['alg']['nenvs'],
        'num_envs_per_env_runner': cfg['alg']['cpu_envs'],
        'num_cpus_per_env_runner': cfg['alg'].get('num_cpus_per_env_runner', 1),
    }
    for key in (
        'rollout_fragment_length',
        'batch_mode',
        'sample_timeout_s',
    ):
        if key in cfg['alg']:
            env_runner_kwargs[key] = cfg['alg'][key]

    resource_kwargs = {
        'num_gpus': cfg['alg'].get('num_gpus', 0),
        'num_cpus_for_main_process': cfg['alg'].get('num_cpus_for_main_process', 1),
    }

    algo_config = (algo
            .environment(env=env_name,
                        #env_config={'num_agents':3},
                        )
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .framework("torch")
            .env_runners(**env_runner_kwargs)
            .resources(**resource_kwargs)
            .multi_agent(policy_mapping_fn=policy_mapping_fn,
                            policies_to_train=policy_training_fn,
                            policies=policy_info)
            .training(**training_kwargs)
            .callbacks(callbacks_class=LogRawEpisodeReturn)
    )

    if 'ppo' in cfg['alg']['type']:
        algo_config.minibatch_size = cfg['alg'].get('minibatch_size', 512)
        algo_config.num_epochs = cfg['alg'].get('num_epochs', 10)
    
    test_env.close()
    del test_env

    return algo_config

def marl_policy_mapping_fn(agent_id, episode, **kwargs):

    return 'agent0' #for single policy, multiple agents

    #return agent_id
