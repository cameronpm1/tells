import copy
import scipy
import numpy as np
from typing import Dict, Optional, Any, Tuple

from controllers.boat_mpc import boatMPC

from tells_environment_dynamics.sim.boat import Boat


class USVGame():

    def __init__(
            self,
            config: Dict[str,Any],
            boats: Dict[str,Boat],
            seed: int = 0,
            board_size: int = 1000,
            
    ):
        '''
        Unmaned Surface Vehicle Game class

        input
        -----
        config:Dict[str,Any]
            loaded configuration file
        boats:Dict[str,Boat]
            dicionary of initialized Boat classes, labeled 'chaser{1-N}' and 'target'
        seed:int
            seed for random number generator
        board_size:int
            wall size of world
        '''
        self.boats = boats
        self.timestep = config['timestep']
        self.horizon = config['horizon']
        self.min_obs = config['min_obs']
        self._seed = seed
        self._random = np.random.default_rng(seed=seed)
        self.board_size = board_size

        self.trajectories = {}
        for key in self.boats.keys():
            self.trajectories[key] = []

        #initialize boat controllers
        '''
        compute discretized state and control matrices using block matrix
        trick from https://en.wikipedia.org/wiki/Discretization#cite_note-2
        '''
        block = np.zeros((9,9))
        block[0:6,0:6] = boats['chaser1'].get_local_attr('A')
        block[0:6,6:9] = boats['chaser1'].get_local_attr('B')
        block_d = scipy.linalg.expm(block*self.timestep*self.horizon)

        A_d = block_d[0:6,0:6]
        B_d = block_d[0:6,6:9]

        self.chaser_mpc = boatMPC(
            initial_state = np.zeros((6,)),
            state_matrix = A_d,
            control_matrix = B_d,
            control_bounds = np.array(config['boats']['chaser1']['mpc']['control_range']),
            dt = self.timestep*self.horizon,
            N = config['boats']['chaser1']['mpc']['N'],
        )

        block = np.zeros((9,9))
        block[0:6,0:6] = boats['target'].get_local_attr('A')
        block[0:6,6:9] = boats['target'].get_local_attr('B')

        block_d = scipy.linalg.expm(block*self.timestep*self.horizon)

        A_d = block_d[0:6,0:6]
        B_d = block_d[0:6,6:9]

        self.target_mpc = boatMPC(
            initial_state = np.zeros((6,)),
            state_matrix = A_d,
            control_matrix = B_d,
            control_bounds = np.array(config['boats']['target']['mpc']['control_range']),
            dt = self.timestep*self.horizon,
            N = config['boats']['target']['mpc']['N'],
        )

        init_function_name = config['sim']['starting_config'] + '_init'
        self.init_function = getattr(self, init_function_name)

        self.reset(init_params=config['sim']['init_params'],state_data=None)

    def reset(
            self,
            init_params: Optional[Dict[str,Any]],
            state_data: Optional[Dict[str,Any]] = None,
    ):
        '''
        reset the states of each USV in the game
        
        input
        -----
        state_data:Optional[Dict[str,Any]]
            dictionary with same keys as self.boat, items are initial state 
            data dictionary (see boat_dynamics)
        '''

        if state_data is None:
            self.init_function(**init_params)
        else:
            for key in self.boats.keys():
                self.boats[key].reset(state_data[key])

        for key in self.boats.keys():
            self.trajectories[key] = []
            for i in range(self.min_obs):
                self.trajectories[key].append(self.boats[key].get_local_attr('pos'))
            

    def forward_step(
            self,
    ) -> None:
        '''
        forward dynamic step for all boats
        '''

        for key in self.boats.keys():
            self.boats[key].forward_step()
            self.trajectories[key].append(self.boats[key].get_local_attr('pos'))


    def circle_init(
            self,
            min_dist: float,
            max_dist: float,
    ):
        '''
        initialize chaser boats in larger circle around target
        
        input
        -----
        min_dist:float
            minimum distance chaser boat can be initialized from target
        max_dist:float
            maximum distance chaser boat can be initialized from target
        '''
        n_boats = len(self.boats) - 1

        temp_state_data = {
            'velocity' : np.array([0.0, 0.0]),
            'heading' : np.array([0.0]),
            'angular_velocity' : np.array([0.0])
        }
        target_start = self.rand_normal(max_dist,self.board_size-max_dist,(2,))
        temp_state_data['position'] = target_start
        self.boats['target'].reset(temp_state_data)

        for key in self.boats.keys():
            if 'chaser' in key:
                temp_state_data = {
                    'velocity' : np.array([0.0, 0.0]),
                    'heading' : np.array([0.0]),
                    'angular_velocity' : np.array([0.0])
                }

                idx = int(key[-1]) - 1 #chaser number
                spacing = np.pi*2/n_boats
                angle = self.rand_normal(idx*spacing,(idx+1)*spacing,(1,))[0]
                rad = self.rand_normal(min_dist,max_dist,(1,))[0]
                chaser_start = np.array([rad*np.cos(angle),rad*np.sin(angle)])
                temp_state_data['position'] = chaser_start + target_start
                self.boats[key].reset(temp_state_data)

        self.target_goal = target_start

    def furthest_from_centroid(
            self,
            window_size: float = 800
    ) -> list[float]:
        '''
        return point within map that is furthest from the centroid of 
        the chaser boat craft given it is inside the window of interest

        input
        -----
        window_size:float
            size of window of interest (centered at target position)

        output
        ------
        list[float] point on the map furthest from the centroid
        '''

        n_boats = len(self.boats) - 1
        centroid = np.zeros((2,))

        for key in self.boats.keys():
            if 'chaser' in key:
                centroid = centroid + self.boats[key].get_local_attr('pos')

        centroid = centroid/n_boats

        target_pos = self.boats['target'].get_local_attr('pos')
        #board_corners = np.array([
        #    [target_pos[0]-(window_size/2),target_pos[1]-(window_size/2)],
        #    [target_pos[0]-(window_size/2),target_pos[1]+(window_size/2)],
        #    [target_pos[0]+(window_size/2),target_pos[1]-(window_size/2)],
        #    [target_pos[0]+(window_size/2),target_pos[1]+(window_size/2)],
        #])
        board_corners = np.array([
            [0.0,0.0],
            [self.board_size, 0.0],
            [self.board_size, self.board_size],
            [0.0, self.board_size]
        ])
        board_corners = np.clip(board_corners,[0,0],[self.board_size,self.board_size])
        furthest_point = board_corners[np.argmax(np.linalg.norm(board_corners - centroid, axis=1))]

        return furthest_point
    
    def centroid_distribution(
            self,
    ) -> Tuple[float,float]:
        '''
        return statistics on controid of chaser boats

        output
        ------
        tuple[float,float]
            the average point and standard deviation of the centroid
        '''

        n_boats = len(self.boats) - 1
        centroid = []

        for key in self.boats.keys():
            if 'chaser' in key:
                centroid.append(self.boats[key].get_local_attr('pos'))

        centroid_avg = np.average(centroid,axis=0)
        centroid_std = np.std(centroid, axis=0)
        
        return centroid_avg, centroid_std
    
    def target_boat_PFA(
            self,
            chaser_prox: float = 400.0,
            vector_scale: float = 100.0,
            attract_scale: float = 1.0,
            repel_scale: float = 5.0,
            buffer: float = 40,
    ) -> list[float]:
        '''
        Potential Field Algorithm (PFA) for target boat, takes goal point and computes
        appropriate target point given location of chaser boats and walls

        input
        -----
        chaser_prox:float
            proximity to chaser boat for which repulsive force is used
        vector_scale:float
            value to compute PFA computed target vector by to compute target point
        attract_scale:float
            value to scale attractive vectors
        repel_scale:float
            value to scale repulsive vectors
        buffer:float
            closest distance goal position can be from a wall

        output
        ------
        list[float]
            goal point for target_boat
        '''

        vec = np.zeros((2,))
        target_pos = self.boats['target'].get_local_attr('pos')

        goal = self.furthest_from_centroid()
        goal_vec = goal - target_pos
        norm_goal_vec = goal_vec/np.linalg.norm(goal_vec) 
        vec = vec + (norm_goal_vec * attract_scale)

        for key in self.boats.keys():
            if 'chaser' in key:
                repel_vec = target_pos - self.boats[key].get_local_attr('pos')
                norm_repel_vec = repel_vec/np.linalg.norm(repel_vec)
                weight = self.chaser_boat_potential_function(distance=np.linalg.norm(repel_vec)/chaser_prox) * repel_scale
                vec = vec + (norm_repel_vec * weight)

        scaled_vec = vec/np.linalg.norm(vec) * vector_scale
        target_point = target_pos + scaled_vec

        return np.clip(target_point,[buffer,buffer],[self.board_size-buffer,self.board_size-buffer])

    def chaser_boat_potential_function(
            self,
            distance: float,
            scale: float = 1.0
    ) -> float:
        '''
        potential function for weight of a chaser boat to be used in PFA 
        modified sigmoid function: 1 / (1 + e^(10x-5))

        input
        -----
        distance:float
            normalized distance between chaser boat and target boat (assumes in range [0,1])
            returns 1, 0 for distance values outside of this range
        scale:float
            value to scale function by

        output
        ------
        float
            weight for chaser craft given distance x
        '''

        return scale * 1 / (1 + np.exp((10*distance)-5)) #+ ( 1 / (1 + np.exp(distance)) )
    
    def set_position_control(
            self,
            name: str,
            goal_point: list[float],
    ) -> list[float]:
        '''
        sets control for boat given goal point

        input
        -----
        name:str
            name of boat from boat list to compute control of
        list[float]:goal_point
            goal point for target boat

        output
        ------
        list[float]
            control output for target boat
        '''
            
        goal_state = np.zeros((6,))
        goal_state[0:2] = goal_point

        if 'target' in name:
            control = self.target_mpc.compute_action(
                initial_state = self.boats[name].get_local_attr('state'),
                reference_state = goal_state,
            )
        else:
            control = self.chaser_mpc.compute_action(
                initial_state = self.boats[name].get_local_attr('state'),
                reference_state = goal_state,
            )

        self.boats[name].set_ctrl([control[0],0,control[2]])

        return control


    def rand_normal(
            self,
            low: float,
            high: float,
            size: Tuple = (1,),
    ):
        '''
        return clipped random value using class RNG, mu and sigma computed off of
        median of low and high, sigma set to 1/6th of range
        
        input
        -----
        low:float
            minimum value in range
        high:float
            maximum value in range
        size:tuple
            size of array of random variables to return 
        '''
        mu = (high-low)/2 + low
        sigma = (high-low)/6
        vals = np.clip(self._random.normal(mu,sigma,size),low,high)

        return vals
    
    def get_plot_data(self) -> dict:
        '''
        returns dictionary w/ plottind data for Renderer2D

        output
        ------
        dict
            plotting data keys:[lines,points]
        '''

        boat_points = np.array([
            [0.0, 0.25],
            [1.5, 0.25],
            [2.0, 0.0],
            [1.5, -0.25],
            [0.0, -0.25]
        ])

        boat_lines = np.array([
            (0,1), (1,2), (2,3), (3,4), (4,0)
        ])

        points = []
        lines = []

        for i,boat in enumerate(self.boats.keys()):
            pos = self.boats[boat].get_local_attr('pos')
            dcm = self.boats[boat].get_local_attr('dcm')
            transformed_vertices = np.dot(boat_points*10, dcm.T) + pos
            for j in range(len(boat_points)):
                points.append(transformed_vertices[j])
                lines.append((boat_lines[j][0]+(5*i),boat_lines[j][1]+(5*i)))

        plot_data = {}
        plot_data['lines'] = lines
        plot_data['points'] = points

        return plot_data


    