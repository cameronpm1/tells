import os
import sys
import torch
import numpy as np
from types import SimpleNamespace

from pathlib import Path

IC3NET_REPO = '/home/cameron/external/IC3Net'
sys.path.insert(0, str(IC3NET_REPO))

from external.IC3Net.utils import *
from envs.marl.make_env import make_marl_env
from external.IC3Net.models import RNN as PolicyNet
from util.util import mkdir, load_config, save_argb_video, save_rgb_gif


def make_ic3net_args(env, hid_size=256, rnn_type="LSTM"):
    return SimpleNamespace(
        nagents=env.nagents,
        hid_size=hid_size,
        continuous=False,
        dim_actions=env.dim_actions,
        naction_heads=[env.num_actions],
        rnn_type=rnn_type,
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

    model = load_ic3net_policy(checkpoint_dir, env)

def load_ic3net_policy(checkpoint_path, env, device="cpu"):
    device = torch.device(device)

    policy_net = PolicyNet(
        make_ic3net_args(env),
        num_inputs=env.num_inputs,
    ).to(device)

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

    policy_net.load_state_dict(state_dict)

    policy_net.eval()
    return policy_net

def eval_single_episode(
    env,
    cfg: dict,
    algo,
    save_dir: str = "",
    idx: int = 0,
):

    episode_rewards = []
    episode_lengths = []
    images = []


    algo.eval()
    obs = env.reset()
    done = False
    step_count = 0
    nagents = len(cfg['policy_list'])

    total_reward = 0.0
    last_info = {}

    frame = env.render_rgb()
    images.append(frame)

    while not done:
        with torch.no_grad():
            out = policy_net(obs)
            print(out)
            if isinstance(out, tuple):
                out = out[0]

            # assumes output logits: (1, nagents, num_actions)
            actions = torch.argmax(out, dim=-1)

        obs, rewards, done, info = env.step(actions)

        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)

        total_reward += np.sum(rewards)
        step_count += 1
        last_info = info

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)

        frame = env.render_rgb()
        images.append(frame)

    save_file = str(os.path.join(save_dir,str(idx)+'.gif'))
    print('generating video in ' + save_file)
    save_rgb_gif(images,save_file)

    print("\n==== EVAL RESULTS ====")
    print(f"Reward: {episode_rewards[-1]}")
    print(f"Length: {episode_lengths[-1]}")


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