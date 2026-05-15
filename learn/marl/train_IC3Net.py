import os
import sys
import time
import yaml
import hydra
import torch
import shutil
import random
import pickle
import logging
import argparse
import subprocess
import numpy as np
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from ray.tune.logger import pretty_print
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.algorithms.sac.sac_torch_policy import SACTorchPolicy
from ray.rllib.utils.metrics import (
    NUM_ENV_STEPS_SAMPLED,
)

#from logger import getlogger
from learn.marl.train import (
        _load_checkpoint_iteration,
        _find_latest_checkpoint,
        _save_checkpoint,
        _episodes_done,
        _summarize_eval_rows,
        _evaluate_shared_policy,
        _collect_slot_controller_dataset,
        _maybe_pretrain_policy,
        make_ray_config,
        marl_policy_mapping_fn
)
from util.util import mkdir, load_config
from envs.marl.make_env import make_marl_env
from envs.marl.rllib_wrapper import RLLibWrapper
from controllers.marl_slot_controller import compute_slot_actions


'''
Installing IC3Net in externals:

$python3.10 -m venv .venv_ic3net
$source .venv_ic3net/bin/activate

$python -m pip install --upgrade pip setuptools wheel

$cd externals
$git clone https://github.com/IC3Net/IC3Net.git
$cd IC3Net

$cd ic3net-envs
$python setup.py develop
$pip install -e .
$cd ..

change torch in requirements.txt to 1.13.1

$pip install -r requirements.txt
pillow will likely not install
$pip install torch==1.13.1
$pip install "Pillow>=9.5,<11"
$pip install visdom==0.1.4

$pip install numpy gym gymnasium pygame pettingzoo supersuit omegaconf
$pip install mpe2
$pip install "ray[rllib]"

$pip uninstall -y gym gymnasium
$pip install "numpy==1.23.5"
$pip install "gym==0.23.1"

IC3Net/comm.py line 175 replace with: agent_mask = agent_mask * comm_action_mask.double()
IC3Net/ic3net-envs/ic3net_envs/predator_prey_env.py line 304 replace with: grid = list(np.ogrid[tuple(map(slice, idx.shape))])
IC3Net/trainer.py line 139 and 140 replace with: actions = torch.Tensor(np.asarray(batch.action))
                                                 actions = actions.transpose(1, 2).reshape(-1, n, dim_actions)

install all other necessary dependances for environment

'''

def _get_policy_ids(algo):
    if hasattr(algo, "workers") and algo.workers is not None:
        return list(algo.workers.local_worker().policy_map.keys())

    if hasattr(algo, "env_runner_group"):
        worker = algo.env_runner_group.local_env_runner
        if hasattr(worker, "policy_map"):
            return list(worker.policy_map.keys())

    raise RuntimeError("Could not infer RLlib policy IDs from Algorithm object.")


def _get_rllib_policy(algo, cfg):

    policy_id = 'agent0'

    policy = algo.get_policy(policy_id)

    if policy is None:
        available = None
        try:
            available = list(algo.get_weights().keys())
        except Exception:
            pass

        raise RuntimeError(
            f"Could not find RLlib policy '{policy_id}'. "
            f"Available policy/module IDs from get_weights(): {available}"
        )

    return policy_id, policy


def _extract_torch_model_from_policy(policy):
    """
    Returns the most likely pretrained Torch module from an RLlib policy.

    Priority:
        1. policy.model.q_model
        2. policy.model.qmodel
        3. policy.model.model
        4. policy.model
    """

    if not hasattr(policy, "model"):
        raise RuntimeError("RLlib policy does not have a .model attribute.")

    model = policy.model

    for attr in ["q_model", "qmodel", "q_net", "qnet"]:
        if hasattr(model, attr):
            candidate = getattr(model, attr)
            if hasattr(candidate, "state_dict"):
                return candidate, attr

    if hasattr(model, "model") and hasattr(model.model, "state_dict"):
        return model.model, "model"

    if hasattr(model, "state_dict"):
        return model, "policy_model"

    raise RuntimeError("Could not extract a Torch state_dict from RLlib policy.model.")


def export_pretrained_ppo_qmodel(algo, cfg, logdir):
    policy_id, policy = _get_rllib_policy(algo, cfg)
    torch_model, model_attr = _extract_torch_model_from_policy(policy)

    export_dir = os.path.join(logdir, "pretrained")
    os.makedirs(export_dir, exist_ok=True)

    export_path = os.path.join(export_dir, "ppo_qmodel_for_ic3.pt")

    payload = {
        "state_dict": torch_model.state_dict(),
        "policy_id": policy_id,
        "model_attr": model_attr,
        "source": "rllib_ppo_pretrain",
    }

    torch.save(payload, export_path)

    print(f"Exported PPO pretrained model for IC3Net: {export_path}")
    print(f"Policy ID: {policy_id}")
    print(f"Exported model attr: {model_attr}")

    return export_path


def export_pretrained_ppo_experience(algo, cfg, logdir):
    pretrain_cfg = cfg['alg'].get('pretrain', {})
    if not pretrain_cfg or not pretrain_cfg.get('enabled', False):
        raise RuntimeError(
            "Cannot export expert experience dataset because PPO pretraining is not enabled. "
            "Enable cfg['alg']['pretrain']['enabled'] or provide a separate dataset export path."
        )

    episodes = int(pretrain_cfg.get('episodes', 300))
    print(f"Collecting {episodes} expert episodes from the slot controller for IC3Net pretraining...")
    obs_arr, act_arr, ret_arr = _collect_slot_controller_dataset(cfg, episodes=episodes)

    export_dir = os.path.join(logdir, "pretrained")
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, "ppo_pretrain_experience_for_ic3.pt")

    payload = {
        "obs": obs_arr,
        "actions": act_arr,
        "returns": ret_arr,
        "source": "rllib_ppo_pretrain_experience",
    }
    torch.save(payload, export_path)

    print(f"Exported PPO expert experience for IC3Net: {export_path}")
    print(
        f"Dataset samples: {obs_arr.shape[0]}, obs_dim: {obs_arr.shape[1]}, "
        f"action count: {int(act_arr.shape[0])}"
    )

    return export_path


def run_ppo_pretrain_and_export_qmodel(config_path):
    cfg = load_config(config_path)
    cfg["config_path"] = os.path.abspath(config_path)

    logdir = os.path.abspath(cfg["logdir"])
    cfg["logdir"] = logdir

    if not os.path.exists(logdir):
        print("Save directory not found, creating path ...")
        mkdir(logdir)
    else:
        print("Save directory found ...")

    print("current directory:", logdir)

    OmegaConf.save(config=cfg, f=os.path.join(logdir, "config.yaml"))

    algo_config = make_ray_config(cfg)
    algo_build = algo_config.build_algo()

    resume_dir, resume_iter = _find_latest_checkpoint(logdir)

    if resume_dir is not None:
        print(f"Resuming from checkpoint: {resume_dir}")
        algo_build.restore(resume_dir)
        print(f"Restored training iteration: {max(resume_iter, 0)}")

    else:
        initial_checkpoint = cfg["alg"].get("initial_checkpoint")

        if initial_checkpoint:
            initial_checkpoint = os.path.abspath(initial_checkpoint)
            print(f"Initializing from checkpoint: {initial_checkpoint}")
            algo_build.restore(initial_checkpoint)
        else:
            _maybe_pretrain_policy(algo_build, cfg, logdir)


    qmodel_path = export_pretrained_ppo_qmodel(
        algo=algo_build,
        cfg=cfg,
        logdir=logdir,
    )
    
    dataset_path = export_pretrained_ppo_experience(
        algo=algo_build,
        cfg=cfg,
        logdir=logdir,
    )

    save_dir = os.path.join(logdir, "ppo_pretrain_export_checkpoint")
    _save_checkpoint(algo_build, save_dir)

    algo_build.stop()

    return cfg, qmodel_path, dataset_path


def find_ic3net_dir(root_dir, cfg):
    if "ic3net_dir" in cfg:
        path = Path(cfg["ic3net_dir"]).expanduser().resolve()
        if (path / "main.py").exists():
            return path

    candidates = [
        root_dir / "external" / "IC3Net",
        root_dir / "externals" / "IC3Net",
    ]

    for path in candidates:
        if (path / "main.py").exists():
            return path

    raise FileNotFoundError("Could not find IC3Net main.py under external/IC3Net or externals/IC3Net.")


def build_ic3_command(cfg, qmodel_path, dataset_path, root_dir, ic3net_dir):
    ic3 = cfg["ic3"]

    save_path = Path(
        ic3.get(
            "save_path",
            root_dir / "checkpoints" / "ic3net_from_ppo_qmodel.pt",
        )
    ).expanduser().resolve()

    save_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(ic3net_dir / "main.py"),

        "--env_name", str(ic3.get("env_name", "custom_predator_prey")),
        "--nagents", str(ic3.get("nagents", 3)),
        "--nprocesses", str(ic3.get("nprocesses", 4)),
        "--custom_config", str(cfg["config_path"]),
        "--num_epochs", str(ic3.get("num_epochs", 2000)),
        "--epoch_size", str(ic3.get("epoch_size", 10)),
        "--batch_size", str(ic3.get("batch_size", 500)),

        "--hid_size", str(ic3.get("hid_size", 128)),
        "--detach_gap", str(ic3.get("detach_gap", 10)),

        "--lrate", str(ic3.get("lrate", 0.001)),
        "--gamma", str(ic3.get("gamma", 0.99)),
        "--tau", str(ic3.get("tau", 1.0)),

        "--max_steps", str(ic3.get("max_steps", 100)),
        "--nactions", str(ic3.get("nactions", 1)),

        "--recurrent",
        "--rnn_type", str(ic3.get("rnn_type", "LSTM")),

        "--comm_passes", str(ic3.get("comm_passes", 1)),
        "--seed", str(ic3.get("seed", 0)),

    ]

    if ic3.get("load_qmodel", False):
        cmd.extend(["--load_qmodel", str(qmodel_path)])

    pretrain_data_cfg = ic3.get("pretrain_data", None)

    if isinstance(pretrain_data_cfg, bool):
        pretrain_data_path = dataset_path if pretrain_data_cfg else None
    else:
        pretrain_data_path = pretrain_data_cfg

    if pretrain_data_path is None or pretrain_data_path == "":
        pretrain_data_path = dataset_path

    if pretrain_data_path:
        cmd.extend(["--pretrain_data", str(pretrain_data_path)])
        cmd.extend(["--pretrain_epochs", str(ic3.get("pretrain_epochs", 0))])
        cmd.extend(["--pretrain_batch_size", str(ic3.get("pretrain_batch_size", 256))])
        cmd.extend(["--pretrain_lr", str(ic3.get("pretrain_lr", 0.001))])
        cmd.extend(["--pretrain_value_coef", str(ic3.get("pretrain_value_coef", 0.05))])

    cmd.extend([
        "--save", str(save_path),
        "--save_every", str(ic3.get("save_every", 100)),
    ])

    model_type = str(ic3.get("model", "ic3net"))

    if model_type == "ic3net":
        cmd.append("--ic3net")
    elif model_type == "commnet":
        cmd.append("--commnet")
    elif model_type == "independent":
        pass
    else:
        raise ValueError(f"Unknown cfg['ic3']['model']: {model_type}")

    if ic3.get("load", ""):
        cmd.extend(["--load", str(Path(ic3["load"]).expanduser().resolve())])

    return cmd


def run_ic3_training(cfg, qmodel_path, dataset_path=None):
    root_dir = Path(__file__).resolve().parent
    ic3net_dir = find_ic3net_dir(root_dir, cfg)

    cmd = build_ic3_command(
        cfg=cfg,
        qmodel_path=qmodel_path,
        dataset_path=dataset_path,
        root_dir=root_dir,
        ic3net_dir=ic3net_dir,
    )

    log_path = Path(
        cfg["ic3"].get(
            "log_path",
            root_dir / "logs" / "ic3net_qmodel.log",
        )
    ).expanduser().resolve()

    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(root_dir),
            str(ic3net_dir),
            env.get("PYTHONPATH", ""),
        ]
    )
    env["OMP_NUM_THREADS"] = str(cfg["ic3"].get("omp_num_threads", 1))
    env["MKL_NUM_THREADS"] = str(cfg["ic3"].get("mkl_num_threads", 1))

    print("Starting IC3Net training.")
    print(" ".join(cmd))

    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(ic3net_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        return process.wait()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--cfg", dest="config_path", required=True)
    return parser.parse_args()


def train(config_path):
    
    cfg, qmodel_path, dataset_path = run_ppo_pretrain_and_export_qmodel(config_path)

    exit_code = run_ic3_training(
        cfg=cfg,
        qmodel_path=qmodel_path,
        dataset_path=dataset_path,
    )

    if exit_code != 0:
        raise SystemExit(exit_code)
