import gymnasium
import numpy as np
from typing import Optional
from gymnasium import spaces
from collections import OrderedDict

from envs.usv_game import USVGame

class USVGymEnv(gymnasium.Env):

    def __init__(
            self,
            sim: USVGame,
            max_episode_length: int,
            ctrl_scale: list[float],
            reset_kwargs: dict,
            seed: int = 0,
            distance_tol: float = 500,
            action_scaling_type: str = '_clip',
    ):
        '''
        input
        -----
        sim:USVGame
            usv simulation class
        max_episode_length:int
            max episode length
        max_ctrl:list[float]
            MPC controller max control
        reset_kwargs:dict
            sim reset arguments
        seed:float
            seed for environment
        distance_tol:float
            distance tolerance for determining episode termination
        action_scaling_type:str
            clip, scale, or std used for preprocessing position commands, default = clip 
        """
        '''
        
        self.sim = sim
        self.distance_tol = distance_tol
        self.ctrl_scale = ctrl_scale
        self.reset_kwargs = reset_kwargs
        self.max_episode_length = max_episode_length

        self.n_boats = len(self.sim.boats) - 1
        self.action_dim = self.n_boats * 2
        self._seed = seed
        self._random = np.random.default_rng(seed=seed)
        self._episode = 0
        self._step = 0


    @property
    def action_space(
            self,
    ) -> gymnasium.Space:
        return spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)

    @property
    def observation_space(
            self,
    ) -> gymnasium.Space:

        obs = self._get_obs()
        space = {}
        for key, val in obs.items():
            space[key] = spaces.Box(low=-np.inf, high=np.inf, shape=val.shape)

        return spaces.Dict(space)
    
    def reset(
            self, 
            **kwargs
    ) -> None:
        '''
        Reset the simulation

        input
        -----
        kwargs
            arguments to be passed to sim
        '''

        self._step = 0
        self._episode += 1
        if 'reset_kwargs' in kwargs:
            self.sim.reset(kwargs['reset_kwargs'])
        else:
            self.sim.reset(init_params=self.reset_kwargs)
        self.obs = self._get_obs()

        return self.obs, {'episode': self._episode}

    
    def step(self, 
             action
    ) -> None:
        '''
        forward step functino

        input
        -----
        action:list[float]
            positional change for each boat in the team in m
        '''
        self._step += 1

        for n in range(self.n_boats):
            boat_key = 'chaser' + str(n)
            pos_ctrl = self.sim.boats[boat_key].get_local_attr('pos')+(action[n*2:n*2+2]*self.ctrl_scale)
            self.sim.set_position_control(boat_key,pos_ctrl)

        self.target_goal = self.sim.target_boat_PFA()
        self.sim.set_position_control('target',self.target_goal)

        self.sim.forward_step()

        self.obs = self._get_obs()
        rew = self._reward(target_traj=self.obs['target'])
        terminated, truncated = self._end_episode() #end by collision, end by max episode

        return self.obs, rew, terminated, truncated, {'done': (terminated, truncated), 'reward': rew}

    
    def _reward(
            self,
            target_traj: list[list[float]],
            reward_scale: float = 10
    ) -> float:
        '''
        compute reward for individual boat agent

        input
        -----
        target_traj:list[list[float]]
            trajectory of the target boat from the last timestep
        reward_scale:float
            constant to normalize reward by 
        '''

        total_distance = np.sum(np.linalg.norm(target_traj[1:] - target_traj[:-1], axis=1))

        return total_distance/reward_scale
        
        
    def _end_episode(self) -> bool:
        '''
        check for episode termination or termination

        output
        ------
        tuple[bool,bool]
            termination and truncation, terminate if spacecraft get too close
        '''

        centroid_avg, centroid_std = self.sim.centroid_distribution()
        target_pos = self.sim.boats['target'].get_local_attr('pos')

        distance = np.linalg.norm(centroid_avg-target_pos)

        return distance > self.distance_tol or np.max(centroid_std) > self.distance_tol/2, self._step >= self.max_episode_length

    def _get_obs(self) -> OrderedDict:
        """Return observation

        output
        ------
        dict[str,Any]
            mouse state relative to orbit, and EKF observation for last x steps
        """

        obs = OrderedDict()

        for key in self.sim.boats.keys():
            if 'target' in key:
                obs[key] = np.array(self.sim.trajectories[key][-1*self.sim.min_obs:]) / self.sim.board_size
            else:
                obs[key] = np.array(self.sim.trajectories[key][-1]) / self.sim.board_size

        return obs
    
    def get_plot_data(self) -> dict:
        '''
        returns dictionary w/ plottind data for Renderer2D

        output
        ------
        dict
            plotting data keys:[lines,points,goal]
        '''

        plot_data = self.sim.get_plot_data()
        plot_data['goal'] = self.target_goal

        return plot_data
    
    '''
    SCALING FUNCTIONS
    '''

    def _clip(
            self,
            action: list[float],
            max_action: list[float],
    ) -> list[float]:
        return np.multiply(max_action,np.clip(action,a_min=-1,a_max=1))
    
    def _std(
            self,
            action: list[float],
            max_action: list[float],
    ) -> list[float]:
        if np.std(action) > 1:
            return np.multiply(max_action,action/np.std(action))
        else:
            return np.multiply(max_action,action)
    
    def _scale(
            self,
            action: list[float],
            max_action: list[float],
    ) -> list[float]:
        if abs(action).max > 1:
            return np.multiply(max_action,action/np.linalg.norm(action))
        else:
            return np.multiply(max_action,action)