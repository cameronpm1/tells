import gymnasium
import numpy as np
from copy import deepcopy
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
            horizon: int,
            reset_kwargs: dict,
            seed: int = 0,
            obs_noise: float = 0.0,
            distance_tol: float = 500,
            action_scaling_type: str = '_clip',
            eval: bool = False,
    ):
        '''
        input
        -----
        sim:USVGame
            usv simulation class
        max_episode_length:int
            max episode length
        ctrl_scale:list[float]
            value to scale policy control by
        horizon:int
            number of times to take forward steps in each step
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
        self.horizon = horizon
        self.reset_kwargs = reset_kwargs
        self.obs_noise = obs_noise
        self.max_episode_length = max_episode_length
        self.eval = eval

        self.n_boats = len(self.sim.boats) - 1
        self.action_dim = self.n_boats * 2
        self.min_obs = self.sim.min_obs
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

        self.target_goal = self.sim.target_boat_PFA() 
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

        keys, poses = [],[]

        for n in range(self.n_boats):
            boat_key = 'chaser' + str(n)
            pos_ctrl = self.sim.boats[boat_key].get_local_attr('pos')+(action[n*2:n*2+2]*self.ctrl_scale)
            keys.append(boat_key)
            poses.append(pos_ctrl)
        
        self.target_goal = self.sim.target_boat_PFA()

        plotting_data = []
        for i in range(self.horizon):
            for boat_key,pos_ctrl in zip(keys,poses):
                self.sim.set_position_control(boat_key,pos_ctrl)
            self.sim.set_position_control('target',self.target_goal)
            plotting_data.append(self.get_plot_data())

            self.sim.forward_step()

        self.obs = self._get_obs()
        rew = self._reward()
        terminated, truncated = self._end_episode() #end by collision, end by max episode

        return self.obs, rew, terminated, truncated, {'done': (terminated, truncated), 'reward': rew, 'plotting': plotting_data}

    
    def _reward(
            self,
            reward_scale: float = 1
    ) -> float:
        '''
        compute reward for team of boat agents

        input
        -----
        reward_scale:float
            constant to normalize reward by 
        '''
        distance_matrix = np.zeros((self.n_boats,self.n_boats))
        for b in range(self.n_boats):
            boat_key = 'chaser' + str(b)
            for n in range(self.n_boats):
                distance_matrix[n][b] = np.linalg.norm(self.obs[boat_key][0:2]-self.obs['goals'][n]) * 10

        distance_sum = 0
        for n in range(self.n_boats - 1):
            row, col = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            distance_sum += distance_matrix[row][col]
            distance_matrix = np.delete(distance_matrix, row, axis=0)
            distance_matrix = np.delete(distance_matrix, col, axis=1)

        distance_sum += distance_matrix[0][0]
        return 1/np.clip(distance_sum,0.1,np.inf)
        
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

        distance1 = np.linalg.norm(centroid_avg-target_pos)
        distance2 = np.linalg.norm(self.sim.target_start - target_pos)

        if 'circle' in self.sim.init_type:
            terminate = distance1 > self.distance_tol or distance2 > self.distance_tol or np.max(centroid_std) > self.distance_tol/2
        else:
            #shrinking map cutoff
            if self._step < 2:
                self.centroid_range = (np.linalg.norm(centroid_std) - 150) * 1.3
            centroid_cutoff = (self.centroid_range * (100 - np.clip(self._step,0,100)) / 100) + 150
            terminate = distance1 > self.distance_tol or distance2 > self.distance_tol or np.linalg.norm(centroid_std) > centroid_cutoff

        return terminate, self._step >= self.max_episode_length

    def _get_obs(self) -> OrderedDict:
        """Return observation

        output
        ------
        dict[str,Any]
            mouse state relative to orbit, and EKF observation for last x steps
        """

        obs = OrderedDict()
        norm_scale = self.sim.board_size

        for key in self.sim.boats.keys():
            if 'target' in key:
                obs[key] = np.array(self.sim.trajectories[key][-1*self.min_obs:])
                #make pos relative to targets last position
                obs[key][:,0:2] = (obs[key][:,0:2] - self.sim.trajectories['target'][-1][0:2]) / norm_scale
            else:
                obs[key] = np.array(self.sim.trajectories[key][-1])
                #make pos relative to targets last position
                obs[key][0:2] = ((obs[key][0:2] + self._random.normal(0,self.obs_noise,(1,2))) - self.sim.trajectories['target'][-1][0:2]) / norm_scale
                if self.eval:
                    obs['target_goal'] = (self.target_goal - self.sim.trajectories['target'][-1][0:2]) / norm_scale
                    obs[key+'_true'] = np.array(self.sim.trajectories[key][-1*self.min_obs:]) #[-1])
                    obs[key+'_true'][:,0:2] = (obs[key+'_true'][:,0:2] - self.sim.trajectories['target'][-1][0:2]) / norm_scale
                    
        goals = []
        ang_space = np.pi*2/self.n_boats
        for i in range(self.n_boats):
            ang = obs['target'][-1][2] + (ang_space*i)
            goals.append([0.1 * np.cos(ang), 0.1 * np.sin(ang)])
        obs['goals'] = np.array(goals)

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