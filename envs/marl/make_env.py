import yaml
import numpy
from typing import Optional

from envs.marl.rllib_wrapper import RLLibWrapper
from envs.marl.predator_prey_env import PredatorPreyEnv, parallel_env



def make_predator_prey_env(
        config: dict,
        seed: Optional[int] = None,
        wrap: bool = True,
        eval: bool = False,
        render_mode: Optional[str] = None,
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
    )

    if wrap:
        env = RLLibWrapper(env)
    
    return env
