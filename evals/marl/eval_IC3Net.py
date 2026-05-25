import os
import sys
import torch
import numpy as np
from types import SimpleNamespace
try:
    from inspect import getargspec
except:
    print('Cannot import inspect in eval_IC3Net')

from pathlib import Path

IC3NET_REPO = '/home/cameron/tells/external/IC3Net'
sys.path.insert(0, str(IC3NET_REPO))

# Also ensure top-level `utils` module from the IC3Net repo is importable
sys.path.insert(0, '/home/cameron/tells/external/IC3Net')

from utils import *
from envs.marl.make_env import make_marl_env
from external.IC3Net.models import RNN as RNNPolicy
from external.IC3Net.comm import CommNetMLP
from external.IC3Net.action_utils import select_action, translate_action
from util.util import mkdir, load_config, save_argb_video, save_rgb_gif


def make_ic3net_args(env, hid_size=256, rnn_type="LSTM"):
    num_actions = env.num_actions
    if isinstance(num_actions, int):
        num_actions = [num_actions]
    elif isinstance(num_actions, np.ndarray):
        num_actions = num_actions.tolist()

    naction_heads = num_actions

    return SimpleNamespace(
        nagents=env.nagents,
        hid_size=hid_size,
        continuous=False,
        dim_actions=env.dim_actions,
        num_actions=num_actions,
        naction_heads=naction_heads,
        rnn_type=rnn_type,
        # CommNet / IC3Net defaults used by original implementations
        comm_passes=1,
        recurrent=True,
        batch_size=1,
        share_weights=False,
        comm_init='uniform',
        comm_mask_zero=False,
        hard_attn=True,
        commnet=True,
        comm_action_one=False,
        comm_mode='avg',
        mean_ratio=0,
        advantages_per_action=False,
    )

def eval(
        config_dir:str,
        checkpoint_dir:str = None,
        n_runs: int = 1,
    ):

    cfg = load_config(config_dir)

    env = make_marl_env(
        cfg,
        seed=cfg['seed'],
        wrap='ic3net',
        render_mode='rgb_array',
    )

    args = make_ic3net_args(env)
    model = load_ic3net_policy(checkpoint_dir, env, args)

    # run n_runs episodes and save gifs under model_dir/videos
    # If checkpoint_dir is a file path, save videos next to the file (its parent dir)
    if os.path.isfile(checkpoint_dir):
        base_dir = os.path.dirname(checkpoint_dir)
    else:
        base_dir = checkpoint_dir
    save_dir = os.path.join(base_dir, 'videos')
    mkdir(save_dir)

    for i in range(n_runs):
        eval_single_episode(env=env, cfg=cfg, args=args, algo=model, save_dir=save_dir, idx=i)

def load_ic3net_policy(checkpoint_path, env, args, device="cpu"):
    device = torch.device(device)

    # Choose policy class based on checkpoint contents (supports older CommNet checkpoints)
    # Default to RNNPolicy unless checkpoint looks like CommNet
    # We'll instantiate a placeholder for now and replace after inspecting state_dict
    # Load checkpoint first to inspect keys

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,  # needed for old IC3Net checkpoints
    )

    if isinstance(checkpoint, dict):
        if "policy_net" in checkpoint:
            state_dict = checkpoint["policy_net"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # common IC3Net case may use this
            state_dict = checkpoint
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(checkpoint)}")

    # Heuristic: if checkpoint contains 'encoder' or 'C_modules' keys, it's CommNetMLP
    state_keys = set(k for k in state_dict.keys())
    if any(k.startswith('encoder') or k.startswith('C_modules') or k.startswith('f_module') for k in state_keys):
        # If there are multiple heads saved (heads.<i>), infer their output sizes
        head_keys = [k for k in state_dict.keys() if k.startswith('heads.')]
        if head_keys:
            # collect indices
            head_indices = set(int(k.split('.')[1]) for k in head_keys)
            naction_heads = []
            for i in sorted(head_indices):
                wkey = f'heads.{i}.weight'
                if wkey in state_dict:
                    out_dim = state_dict[wkey].shape[0]
                else:
                    out_dim = env.num_actions
                naction_heads.append(out_dim)
            args.naction_heads = naction_heads

        policy_net = CommNetMLP(args, env.num_inputs).to(device)
    else:
        policy_net = RNNPolicy(args, num_inputs=env.num_inputs).to(device)

    policy_net.load_state_dict(state_dict)

    policy_net.eval()
    return policy_net

def eval_single_episode(
    env,
    cfg: dict,
    args,
    algo,
    save_dir: str = "",
    idx: int = 0,
):

    episode_rewards = []
    episode_lengths = []
    images = []

    algo.eval()
    reset_args = getargspec(env.reset).args
    if 'epoch' in reset_args:
        state = env.reset(0)
    else:
        state = env.reset()

    done = False
    step_count = 0
    nagents = len(cfg['policy_list'])

    total_reward = np.zeros(nagents, dtype=np.float64)
    stat = dict()
    info = {}

    # initialize prev hidden state like Trainer.get_episode
    prev_hid = torch.zeros(1, args.nagents, args.hid_size)

    frame = env.render_rgb()
    images.append(frame)

    while not done:
        misc = dict()
        if step_count == 0 and getattr(args, 'hard_attn', False) and getattr(args, 'commnet', False):
            info['comm_action'] = np.zeros(args.nagents, dtype=int)

        if getattr(args, 'recurrent', False):
            if args.rnn_type == 'LSTM' and step_count == 0:
                prev_hid = algo.init_hidden(batch_size=state.shape[0])

            x = [state, prev_hid]
            output = algo(x, info)
        else:
            x = state
            output = algo(x, info)

        if isinstance(output, tuple):
            action_out = output[0]
            if len(output) > 2:
                ret = output[2]
            else:
                ret = None
        else:
            action_out = output
            ret = None

        action = select_action(args, action_out)
        action, actual = translate_action(args, env, action)

        if ret is not None:
            prev_hid = ret

        next_state, rewards, done, info = env.step(actual)

        if getattr(args, 'hard_attn', False) and getattr(args, 'commnet', False):
            info['comm_action'] = action[-1] if not args.comm_action_one else np.ones(args.nagents, dtype=int)
            nfriendly = getattr(args, 'nfriendly', args.nagents)
            stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:nfriendly]
            if getattr(args, 'enemy_comm', False):
                stat['enemy_comm'] = stat.get('enemy_comm', 0) + info['comm_action'][nfriendly:]

        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
        total_reward += rewards
        step_count += 1

        if 'alive_mask' in info:
            misc['alive_mask'] = info['alive_mask'].reshape(rewards.shape)
        else:
            misc['alive_mask'] = np.ones_like(rewards)

        nfriendly = getattr(args, 'nfriendly', args.nagents)
        stat['reward'] = stat.get('reward', 0) + rewards[:nfriendly]
        if getattr(args, 'enemy_comm', False):
            stat['enemy_reward'] = stat.get('enemy_reward', 0) + rewards[nfriendly:]

        max_steps = getattr(args, 'max_steps', getattr(env, 'max_steps', 400))
        done = done or step_count == max_steps

        episode_rewards.append(total_reward.copy())
        episode_lengths.append(step_count)

        frame = env.render_rgb()
        images.append(frame)

        state = next_state
        if done:
            break

    save_file = str(os.path.join(save_dir, str(idx) + '.gif'))
    print('generating video in ' + save_file)
    save_rgb_gif(images, save_file)

    print("\n==== EVAL RESULTS ====")
    print(f"Per-agent Reward: {total_reward}")
    print(f"Mean reward per agent: {np.mean(total_reward)}")
    print(f"Length: {step_count}")


def _compute_ic3net_actions(algo, obs, env, cfg=None, action_fn=None, device=None):
    """
    Computes actions for IC3Net-style evaluation.

    Supports:
        1. A custom action_fn, recommended for original IC3Net code
        2. algo.act(obs)
        3. algo.select_action(obs)
        4. algo.compute_action(obs)
        5. algo(obs), for PyTorch modules

    The returned actions are passed directly into IC3NetWrapper.step(...).
    Your wrapper already handles common IC3Net action formats.
    """

    cfg = cfg or {}

    if device is not None and torch.is_tensor(obs):
        obs_in = obs.to(device)
    else:
        obs_in = obs

    with torch.no_grad():
        if action_fn is not None:
            actions = action_fn(algo, obs_in, env, cfg)

        elif hasattr(algo, "act"):
            actions = algo.act(obs_in)

        elif hasattr(algo, "select_action"):
            actions = algo.select_action(obs_in)

        elif hasattr(algo, "compute_action"):
            actions = algo.compute_action(obs_in)

        elif callable(algo):
            output = algo(obs_in)

            # Some models return (actions, hidden_state), (logits, values), etc.
            if isinstance(output, tuple):
                output = output[0]

            actions = output

        else:
            raise TypeError(
                "Could not compute IC3Net actions. Pass a callable model, an algo with "
                "act/select_action/compute_action, or provide action_fn."
            )

    return actions