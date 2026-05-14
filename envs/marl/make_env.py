import yaml
import numpy
from typing import Optional
from gym_pybullet_drones.utils.enums import ActionType

from envs.marl.drones_env import DronesEnv
from envs.marl.rllib_wrapper import RLLibWrapper
from envs.marl.IC3Net_wrapper import IC3NetWrapper
from envs.marl.predator_prey_env import PredatorPreyEnv, parallel_env

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

    env = DronesEnv(
        agent_list=config['env']['agent_list'],
        learned_agent_list=config['env']['learned_agent_list'],
        gui=False,
        act=ActionType.VEL,
        episode_len_sec=config['env']['max_episode_length']*config['timestep'], # 10Hz step rate
        **config['env']['env_kwargs'],
    )

    if wrap == 'rllib':
        env = RLLibWrapper(env,eval,belief)
    elif wrap == 'ic3net':
        env = IC3NetWrapper(env)
    
    return env