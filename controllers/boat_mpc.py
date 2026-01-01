import time
import scipy
import numpy as np
import gurobipy as gp
from gurobipy import GRB, nlfunc



class boatMPC():

    def __init__(
            self,
            initial_state: list[float],
            state_matrix: list[list[float]],
            control_matrix: list[list[float]],
            control_bounds: list[float],
            dt: float,
            N: int = 10,
    ):
        '''
        MPC class for thrust control using gorubi
        example: https://support.gurobi.com/hc/en-us/community/posts/4406846579089-Formulating-Model-Predictive-Control-in-Gurobi-Python

        input
        -----
        initial_state:list[float]
            initial state of system
        state_matrix:list[list[float]]
            state matrix for propagation
        control_matrix:list[list[float]]
            control matrix for propagation
        control_bounds:list[float]
            control upper bounds in Newtons, assumes control lower bound is -control_bounds
        dt:float
            timestep
        N:int
            horizon
        id:int
            for parallel gurobi envs
        '''

        self.N = N
        self.dt = dt
        #discretize state and control matrix
        self.A = state_matrix[0:4,0:4]
        self.B = control_matrix[0:4,0:2]
        self.control_bounds = control_bounds
        #create weight matrices
        self.Q = np.diag([1,1,0.1,0.1]) #reference error weights
        self.R = np.diag([1,1]) * 0.00001 #fuel consumption weights

        #initialize gurobi optimization problem
        self.m = gp.Model("MPC")
        self.m.Params.LogToConsole = 0 #turn off Gurobi print statements
        self.m.Params.Threads = 1 #reduce thread count

        umin = np.tile(-control_bounds, (N,1))
        umax = np.tile(control_bounds, (N,1))

        self.x = self.m.addMVar(shape=(N+1,len(self.A)), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
        self.z = self.m.addMVar(shape=(N+1,len(self.A)), lb=-GRB.INFINITY, ub=GRB.INFINITY, name='z')
        self.u = self.m.addMVar(shape=(N, 2), lb=umin[:,0:2], ub=umax[:,0:2], name='u')

        #points_x = np.linspace(0, np.pi*2, 100)
        #points_sin = np.cos(points_x)
        #points_cos = np.cos(points_x)

        # Add the piecewise linear constraint: y is the result of applying the PWL function to x
        #self.m.addGenConstrPWL(x, y, points_x, points_y, "pwl_cos_constraint")

        obj1 = sum(self.z[k, :] @ self.Q @ self.z[k, :] for k in range(self.N+1))
        obj2 = sum(self.u[k, :] @ self.R @ self.u[k, :] for k in range(self.N))
        self.m.setObjective(obj1 + obj2, GRB.MINIMIZE)

    def set_control(
            self,
            control_matrix: list[list[float]]
    ):
        '''
        input
        -----
        control_matrix:list[list[float]]
            new control matrix
        '''

        self.B = control_matrix

    def compute_action(
            self,
            initial_state: list[float],
            reference_state: list[float],
    ):
        '''
        computes control to reach reference state

        input
        -----
        initial_state:list[float]
            current state of system to propagate from
        reference_state:list[float]
            state to get to
        '''
        action = np.zeros((3,))
        hdg = initial_state[4]
        dist = np.linalg.norm(initial_state[0:2] - reference_state[0:2])

        # Define vectors
        if dist > 1.0:
            hdg_vec = np.array([np.cos(hdg), np.sin(hdg)])
            ref_vec = reference_state[0:2] - initial_state[0:2]
            delta_hdg = np.arctan2(ref_vec[1], ref_vec[0]) - np.arctan2(hdg_vec[1], hdg_vec[0])
            delta_hdg = np.degrees((delta_hdg + np.pi) % (2 * np.pi) - np.pi)
            if abs(delta_hdg) > 90 and dist < 10:
                delta_hdg = delta_hdg + np.sign(-delta_hdg)*180
        else:
            delta_hdg = 0.0

        if abs(delta_hdg) >=  self.control_bounds[2]:
            new_hdg = hdg + delta_hdg/abs(delta_hdg)*self.control_bounds[2]
            delta_hdg = delta_hdg/abs(delta_hdg)*self.control_bounds[2]
        else:
            new_hdg = hdg + delta_hdg
        self.t = self.get_dcm(hdg=new_hdg)

        rel_initial_state = initial_state[0:4].copy()
        rel_initial_state[0:2] = np.zeros((2,))
        rel_reference_state = reference_state[0:4].copy()
        rel_reference_state[0:2] = rel_reference_state[0:2] - initial_state[0:2]
        thrust_control = self.compute_thrust(rel_initial_state,rel_reference_state)

        action[0:2] = thrust_control
        action[2] = np.radians(delta_hdg)

        return action
        

    def compute_thrust(
            self,
            initial_state: list[float],
            reference_state: list[float],
    ):
        '''
        computes thrust control to reach reference state

        input
        -----
        initial_state:list[float]
            current state of system to propagate from
        reference_state:list[float]
            state to get to
        '''
        self.m.remove(self.m.getConstrs())

        self.m.addConstr(self.x[0, :] == initial_state)

        for k in range(self.N):
            #compute errors
            self.m.addConstr(self.z[k, :] == self.x[k, :] - reference_state)
            self.m.addConstr(self.x[k+1, :] == self.A @ self.x[k, :] + self.B @ (self.t @ self.u[k, :]) )

        self.m.addConstr(self.z[self.N, :] == self.x[self.N, :] - reference_state)

        #optimize and return first computed control
        self.m.optimize()
        return self.u[0].X
    
    def get_dcm(
            self,
            hdg: float
        ) -> list[list[float]]:
        '''
        input
        -----
        hdg:float
            heading of vehicle
        output
        ------
        list[list[float]]
            2x2 DCM computed from heading (rotation around z-axis)
        '''

        dcm = np.array([
            [np.cos(hdg), -np.sin(hdg)],
            [np.sin(hdg),  np.cos(hdg)]
        ])

        return dcm
    

    
     