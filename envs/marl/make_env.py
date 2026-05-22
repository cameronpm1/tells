import yaml
import numpy
import sys
from typing import Optional
from pathlib import Path

from envs.marl.rllib_wrapper import RLLibWrapper
from envs.marl.IC3Net_wrapper import IC3NetWrapper

def make_marl_env(
        config: dict,
        seed: Optional[int] = None,
        wrap: Optional[str] = 'rllib',
        eval: bool = False,
        render_mode: Optional[str] = None,
        belief: bool = False,
):
    '''
    initialize marl env

    input
    -----
    config_dir:dict
        config dictionary
    seed:int
        seed
    wrap:bool
        if True wraps env in rllib wrapper
    '''

    env_name = config['env']['scenario']

    if 'predator_prey' in env_name:
        return make_predator_prey_env(config,seed,wrap,eval,render_mode,belief)
    
    elif 'drones' in env_name:
        return make_drones_env(config,seed,wrap,eval,render_mode,belief)
    
    else:
        raise NotImplementedError(f'Env {env_name} not implemented yet')

def make_predator_prey_env(
        config: dict,
        seed: Optional[int] = None,
        wrap: Optional[str] = 'rllib',
        eval: bool = False,
        render_mode: Optional[str] = None,
        belief: bool = False,
):
    '''
    initialize usv_gym_env

    input
    -----
    config_dir:dict
        config dictionary
    seed:int
        seed
    wrap:bool
        if True wraps env in rllib wrapper
    '''
    from envs.marl.predator_prey_env import PredatorPreyEnv, parallel_env

    scenario_kwargs = config['env']['scenario_kwargs']
    scenario_kwargs['agent_list'] = config['env']['agent_list']
    env = parallel_env(
        scenario_kwargs=scenario_kwargs,
        max_cycles=config['env']['max_episode_length'],
        render_mode = render_mode,
    )

    env = PredatorPreyEnv(
        mpeEnv=env,
        agents=config['env']['learned_agent_list'],
        reward_kwargs=config['env'].get('reward_kwargs'),
        controller_kwargs=config['env'].get('controller_kwargs'),
        seed = seed,
        local_observations = config['env']['local_observations'],
    )

    if wrap == 'rllib':
        env = RLLibWrapper(env,eval,belief)
    elif wrap == 'ic3net':
        env = IC3NetWrapper(env)
    
    return env

def make_drones_env(
        config: dict,
        seed: Optional[int] = None,
        wrap: Optional[str] = 'rllib',
        eval: bool = False,
        render_mode: Optional[str] = None,
        belief: bool = False,
):
    '''
    initialize drones_env using pybullet-drones

    input
    -----
    config_dir:dict
        config dictionary
    seed:int
        seed
    wrap:bool
        if True wraps env in rllib wrapper
    '''
    pybullet_drones_path = Path(__file__).resolve().parents[2] / "external" / "pybullet-drones"
    if pybullet_drones_path.exists() and str(pybullet_drones_path) not in sys.path:
        sys.path.insert(0, str(pybullet_drones_path))

    from gym_pybullet_drones.utils.enums import ActionType
    from envs.marl.drones_env import DronesEnv

    env_kwargs = dict(config['env']['env_kwargs'])
    ctrl_freq = int(env_kwargs.get('ctrl_freq', 30))
    episode_len_sec = env_kwargs.pop(
        'episode_len_sec',
        float(config['env']['max_episode_length']) / float(ctrl_freq),
    )

    env = DronesEnv(
        agent_list=config['env']['agent_list'],
        learned_agent_list=config['env']['learned_agent_list'],
        reward_kwargs=config['env'].get('reward_kwargs'),
        controller_kwargs=config['env'].get('controller_kwargs'),
        gui=False,
        act=ActionType.VEL,
        # PyBullet-drones measures episode length in seconds; our config keeps
        # the old RL convention where max_episode_length means control steps.
        episode_len_sec=episode_len_sec,
        seed=seed,
        **env_kwargs,
    )

    if wrap == 'rllib':
        env = RLLibWrapper(env,eval,belief)
    elif wrap == 'ic3net':
        env = IC3NetWrapper(env)
    
    return env
