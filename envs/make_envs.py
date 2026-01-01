import yaml
import numpy
from gymnasium.wrappers import FlattenObservation, FilterObservation

from envs.usv_game import USVGame
from envs.usv_gym_env import USVGymEnv
from tells_environment_dynamics.sim.boat import Boat
from tells_environment_dynamics.sim.boat_dynamics import boatDynamics

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
            return params
        except yaml.YAMLError as exc:
            print(exc)

def make_usv_game(
        config: dict
):
    '''
    initialize usv_game class

    input
    -----
    config:dict
        configuration dictionary
    '''

    boats = {}

    for boat in config['boats']:
        dynamics = boatDynamics(
            inertial_data=config['boats'][boat]['inertial_data'],
            initial_state_data=config['boats'][boat]['initial_state_data'],
            timestep=config['timestep'],
            horizon=config['horizon'],
        )

        #initialize satellite container class
        usv = Boat(
            name=boat,
            dynamics=dynamics,
        )

        boats[boat] = usv

    game = USVGame(
        config = config,
        boats = boats,
        seed = config['seed']
    )

    return game

def make_usv_env(
        config: dict,
        wrap: bool = True,
):
    '''
    initialize usv_gym_env

    input
    -----
    config_dir:dict
        config dictionary
    wrap:bool
        if True wraps env in gymnasium flattenObservation wrapper
    '''

    usv_game = make_usv_game(config)

    env_kwargs = config['env']
    env_kwargs['sim'] = usv_game
    env_kwargs['reset_kwargs'] = config['sim']['init_params']
    env = USVGymEnv(**env_kwargs)

    if wrap:
        env = FilterObservation(env,filter_keys=list(config['boats'].keys()))
        env = FlattenObservation(env)
    
    return env


