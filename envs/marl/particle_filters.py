import time
import numpy as np
from typing import List
from copy import deepcopy
import matplotlib.pyplot as plt

from controllers.fire_control import ACTION_LIBRARY

class Particle:
    def __init__(self, 
        position, 
        velocity_dir, 
        speed, 
        dim, 
        max_speed,
    ):

        self.dim = dim
        self.max_speed = max_speed
        self.position = np.array(position, dtype=float)
        self.velocity_dir = np.array(velocity_dir, dtype=float)
        self.speed = np.clip(speed,0,self.max_speed)

        # Normalize velocity direction
        norm = np.linalg.norm(self.velocity_dir)
        if norm == 0:
            raise ValueError("Velocity direction vector cannot be zero")
        self.velocity_dir /= norm

    def add_control(self, rel_vel, scale_mag=0.1):
        '''
        changes particle velocity vector by rel_velocity,
        '''

        if np.linalg.norm(rel_vel) > 0:
            change_vel = rel_vel/np.linalg.norm(rel_vel)

            new_velocity = self.velocity_dir + change_vel
            self.speed = np.linalg.norm(new_velocity)
            self.velocity_dir = new_velocity/self.speed
            self.speed = np.clip(self.speed,0,self.max_speed)

        #self.velocity = rel_vel
        #self.speed = np.clip(np.linalg.norm(self.velocity),0,self.max_speed)

    def set_position(self, position):

        self.position = position
    
    def propagate(self, dt, process_noise_std_pos=0.01, process_noise_std_dir=0.03):
        """
        Propagate particle state forward by dt using simple kinematic model.

        Args:
            dt (float): Time step
            process_noise_std_pos (float or tuple): Std dev for position noise.
                If float, same std for all dimensions.
                If tuple, must match dim.
            process_noise_std_dir (float): Std dev for velocity direction noise (radians)
                For 3D, this will be applied as a small random rotation.
        """
        if self.dim == 2:

            # 2D case: rotate velocity_dir by small angle noise
            angle_noise = np.random.normal(0, process_noise_std_dir)
            c, s = np.cos(angle_noise), np.sin(angle_noise)
            rot_matrix = np.array([[c, -s], [s, c]])
            self.velocity_dir = rot_matrix @ self.velocity_dir
            self.velocity_dir = self.velocity_dir / np.linalg.norm(self.velocity_dir)

            # Position noise
            noise = np.random.normal(0, process_noise_std_pos, size=2)
            
            # Update position

            displacement = self.velocity_dir * self.speed * dt
            displacement = displacement + noise
            self.position = self.position + displacement

        
        elif self.dim == 3:
            # 3D case: apply small random rotation to velocity_dir
            # Generate a small random rotation axis and angle
            axis = np.random.normal(0, 1, size=3)
            axis /= np.linalg.norm(axis)
            angle = np.random.normal(0, process_noise_std_dir)
            
            # Rodrigues' rotation formula
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)
            
            self.velocity_dir = R @ self.velocity_dir
            self.velocity_dir /= np.linalg.norm(self.velocity_dir)
            
            noise = np.random.normal(0, process_noise_std_pos, size=3)
            
            # Update position
            displacement = self.velocity_dir * self.speed * dt
            self.position += displacement + noise
        
        else:
            raise ValueError("Unsupported dimension: only 2 or 3 allowed")

class ParticleCluster:
    def __init__(self, 
        control_function,
        num_particles: int = 100,
        mean_pos: list[float] = [0.0,0.0],
        std_dev: list[float] = [0.0,0.0],
        resample_threshold: float = 3.0,
        dim: int = 2,
        max_speed: float = 0.4,
        dt: float = 0.1,
        target = False,
    ):
        """
        Args:
            num_particles (int): number of particles in the cluster
        """

        self.num_particles = num_particles
        self.dim = dim
        self.max_speed = max_speed
        self.dt = dt
        self.target = target
        self.particles = []
        self.weights = np.ones(num_particles) / num_particles
        self.resample_threshold = resample_threshold

        self.control_func = control_function
        self.initialize_gaussian(mean_pos,std_dev)
    
    def initialize_gaussian(self, mean_pos, std_dev):

        del self.particles

        self.particles = []
        mean_pos = np.array(mean_pos)
        
        for _ in range(self.num_particles):
            position = np.random.normal(mean_pos, std_dev)
            
            # Random unit velocity direction
            angle = np.random.uniform(0, 2*np.pi)
            if self.dim == 2:
                velocity_dir = np.array([np.cos(angle), np.sin(angle)])
            else:
                velocity_dir = np.array([np.cos(angle), np.sin(angle), 0.0])
            speed = 0.0  # all speeds set to zero
            
            self.particles.append(Particle(position, velocity_dir, speed, self.dim, self.max_speed))

        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def propagate(self, dt, current_obs, target, team_positions, obs_map):

        num_particles = len(self.particles)

        for i, particle in enumerate(self.particles):
            #convert to relative observations
            temp_obs = deepcopy(current_obs)
            team_positions_i = np.concatenate((team_positions,current_obs[obs_map['self_pos']]))
            team_positions_i = team_positions_i - np.tile(particle.position,len(team_positions_i)//self.dim)
            target_i = target - particle.position
            temp_obs[obs_map['target_goal']] = temp_obs[obs_map['target_goal']] + np.tile(current_obs[obs_map['self_pos']]-particle.position,len(temp_obs[obs_map['target_goal']])//self.dim)
            temp_obs[obs_map['self_pos']] = particle.position
            temp_obs[obs_map['target_pos']] = target_i
            if not self.target:
                temp_obs[obs_map['team']] = team_positions_i
            else:
                temp_obs[obs_map['target_obs']] = team_positions_i

            vel_cmd = self.control_func(temp_obs,obs_map)

            particle.add_control(vel_cmd)
            particle.propagate(dt)

    
    def resample(self):
        """
        Resample particles according to their weights using systematic resampling.
        """

        positions = np.array([p.position for p in self.particles])
        idxs = list(range(0,self.num_particles))
        new_pos_idxs = np.random.choice(idxs, size=self.num_particles, p=self.weights)
        self.weights.fill(1.0 / self.num_particles)

        for p,idx in zip(self.particles,new_pos_idxs):
            p.set_position(deepcopy(positions[idx]))
    
    def estimate_mean_position(self):
        """
        Compute weighted average position of the cluster.

        Returns:
            np.array: estimated 2D position
        """
        positions = np.array([p.position for p in self.particles])
        mean_pos = np.average(positions, axis=0, weights=self.weights)
        return mean_pos, 1/np.max(np.std(positions,axis=0)) #np.average(np.stdev(positions, axis=0))

    def estimate_direction(self):
        """
        Compute weighted average heading (unit vector) of the cluster.

        Returns:
            np.array: estimated unit direction vector
        """
        velocities = np.array([p.velocity_dir * p.speed for p in self.particles])
        mean_vel = np.average(velocities, axis=0, weights=self.weights)
        norm = np.linalg.norm(mean_vel)
        if norm == 0:
            return mean_vel
        return mean_vel / norm

    def update_weights(self, measurement, measurement_std):
        """
        Update particle weights based on a measurement (e.g., observed position).

        Args:
            measurement (np.array): observed 2D position
            measurement_std (float): standard deviation of measurement noise
        """

        positions = np.array([p.position for p in self.particles])
        dist = np.linalg.norm(positions - measurement, axis=1)
        if min(dist) > self.resample_threshold:
            self.initialize_gaussian(measurement,np.ones(measurement.shape)*measurement_std)
            return True
        likelihoods = np.exp(-0.5 * dist**2) 
        self.weights = likelihoods + 1e-300  
        self.weights /= np.sum(self.weights)
        return False

    def get_state(self):

        positions = np.array([p.position for p in self.particles])
        velocities = np.array([p.velocity_dir * p.speed for p in self.particles])
        return positions, velocities


class FootballParticleCluster(ParticleCluster):
    def __init__(self,
        control_function,
        num_particles: int = 100,
        mean_pos: list[float] = [0.0,0.0],
        std_dev: list[float] = [0.0,0.0],
        resample_threshold: float = 0.06,
        pass_threshold: bool = 0.15,
        dim: int = 2,
        max_speed: float = 0.4,
        dt: float = 0.1,
        target = False,
        process_noise_std_pos: float = 0.001,
        process_noise_std_dir: float = 0.01,
    ):
        super().__init__(
            control_function,
            num_particles=num_particles,
            mean_pos=mean_pos,
            std_dev=std_dev,
            resample_threshold=resample_threshold,
            dim=dim,
            max_speed=max_speed,
            dt=dt,
            target=target,
        )

        self.pass_threshold = pass_threshold
        self.process_noise_std_pos = process_noise_std_pos
        self.process_noise_std_dir = process_noise_std_dir

    def propagate(self, dt, current_obs, target, team_positions, ball_owner, ball_position, obs_map):

        num_particles = len(self.particles)

        pass_count = 0
        self.last_vel_cmds = []

        for i, particle in enumerate(self.particles):
            #convert to relative observations
            temp_obs = deepcopy(current_obs)
            team_positions_i = team_positions - np.tile(particle.position,len(team_positions)//self.dim)
            temp_obs[obs_map['self_pos']] = particle.position
            temp_obs[obs_map['target_pos']] = target['pos'] - particle.position
            temp_obs[obs_map['target_vel']] = target['vel']
            temp_obs[obs_map['ball_owner']] = ball_owner
            if not self.target:
                temp_obs[obs_map['team']] = team_positions_i
            else:
                temp_obs[obs_map['target_obs']] = team_positions_i
                temp_obs[obs_map['ball_pos']] = ball_position - particle.position
                temp_obs[obs_map['target_ball']] = ball_position - particle.position

            vel_cmd = self.control_func(temp_obs,obs_map)
            self.last_vel_cmds.append(vel_cmd)

            if vel_cmd is None:
                pass_count += 1
                vel_cmd = np.zeros((2,))

            particle.add_control(vel_cmd)
            particle.propagate(dt, process_noise_std_pos=self.process_noise_std_pos, process_noise_std_dir=self.process_noise_std_dir)

        self.last_pass_fraction = pass_count/num_particles

        if pass_count/num_particles > self.pass_threshold:
            return 1 #pass
        else:
            return 0 #no pass

class FireDroneParticleCluster(ParticleCluster):
    def __init__(self,
        control_function,
        num_particles: int = 100,
        mean_pos: list[float] = [0.0,0.0],
        std_dev: list[float] = [0.0,0.0],
        resample_threshold: float = 50,
        extinguish_threshold: float = 0.3,
        dim: int = 2,
        max_speed: float = 1.0,
        dt: float = 1.0,
        target = False,
        process_noise_std_pos: float = 0.001,
        process_noise_std_dir: float = 0.01,
    ):
        super().__init__(
            control_function,
            num_particles=num_particles,
            mean_pos=mean_pos,
            std_dev=std_dev,
            resample_threshold=resample_threshold,
            dim=dim,
            max_speed=max_speed,
            dt=dt,
            target=target,
        )

        self.extinguish_threshold = extinguish_threshold
        self.process_noise_std_pos = process_noise_std_pos
        self.process_noise_std_dir = process_noise_std_dir

    def propagate(self, dt, target, team_positions, name):

        num_particles = len(self.particles)

        extinguish_count = 0
        self.last_vel_cmds = []
        obs = {}
        obs['target'] = deepcopy(target)
        for agent in team_positions.keys():
            if name != agent:
                obs[agent] = {}
                obs[agent]['self_pos'] = deepcopy(team_positions[agent])

        for i, particle in enumerate(self.particles):
            temp_obs = deepcopy(obs)
            temp_obs[name] = {'self_pos': particle.position}

            action = self.control_func(temp_obs)[name]

            if action == 9:
                extinguish_count += 1
                vel_cmd = np.zeros((2,))
            else:
                vel_cmd = ACTION_LIBRARY[action].astype(float)

            particle.add_control(vel_cmd)
            particle.propagate(dt, process_noise_std_pos=self.process_noise_std_pos, process_noise_std_dir=self.process_noise_std_dir)

        if extinguish_count/num_particles > self.extinguish_threshold:
            return 1 #extinguish
        else:
            return 0 #no pass



class PredatorPreyParticleFilter:
    def __init__(
        self, 
        obs_map,
        agent_name,
        agent_start_pos: dict,  # e.g., {'agent0': [x0, y0], 'agent1': [x1, y1]}
        target_start_pos: list[float], 
        agent_control_function,
        target_control_function,
        num_particles: int = 100,
        resample_threshold: float = 8.0,
        std_dev: float = 0.2,
        max_speed: float = 1.0,
        speed_ratio: float = 0.4,
        dt: float = 0.1
    ):
        self.obs_map = obs_map
        self.agent_name = agent_name
        self.n_agents = len(agent_start_pos.keys())
        self.target_pos = np.array(target_start_pos, dtype=float)
        self.clusters = {}
        self.dim = len(target_start_pos)
        self.dt = dt
        self.max_speed = max_speed
        self.speed_ratio = speed_ratio
        self.std_dev = std_dev

        for i, (name, pos) in enumerate(agent_start_pos.items()):
            mean_pos = np.array(pos, dtype=float)
            cluster = ParticleCluster(
                agent_control_function,
                num_particles=num_particles,
                mean_pos=mean_pos,
                std_dev=np.ones(self.dim) * self.std_dev,
                resample_threshold=resample_threshold,
                dim=self.dim,
                max_speed=self.max_speed,
                dt=dt,
                target=False,
            )
            self.clusters[name] = cluster

        # Initialize prey cluster
        self.clusters['target'] = ParticleCluster(
            target_control_function,
            num_particles=num_particles,
            mean_pos=self.target_pos,
            std_dev=np.ones(self.dim) * self.std_dev,
            resample_threshold=resample_threshold,
            dim=self.dim,
            max_speed=self.max_speed*self.speed_ratio,
            dt=dt,
            target=True,
        )


    def reset(
        self,
        obs: dict,  # e.g., {'agent0': [x0, y0], 'agent1': [x1, y1]}
    ):
        # Initialize one ParticleCluster per predator
        for name in self.clusters.keys():
            mean_pos = np.array(obs[name][self.obs_map['self_pos']], dtype=float)
            self.clusters[name].initialize_gaussian(mean_pos,np.ones(self.dim) * self.std_dev)

    def propagate_all(self, current_obs):

        team_positions = []
        target_position = None

        # Get estimated mean positions
        for name in self.clusters.keys():
            mean_pos, _ = self.clusters[name].estimate_mean_position()
            if 'agent' in name:
                team_positions.append(mean_pos)
            elif name == 'target':
                target = mean_pos

        team_positions = np.array(team_positions)

        # Propagate predator clusters
        for i, name in enumerate(self.clusters.keys()):
            if 'agent' in name:
                self.clusters[name].propagate(self.dt, current_obs, target, np.delete(deepcopy(team_positions), i, axis=0).flatten(), self.obs_map)
            else:
                self.clusters[name].propagate(self.dt, np.concatenate((current_obs,np.zeros((self.dim,)))), target, team_positions.flatten(), self.obs_map)

    def update_observation(
        self, 
        agent_name: str, 
        observed_pos: np.ndarray, 
        measurement_std: float = 0.1
    ):

        if agent_name not in self.clusters:
            raise ValueError(f"Agent name '{agent_name}' not found in clusters.")

        force_reset = self.clusters[agent_name].update_weights(observed_pos, measurement_std)
        if not force_reset:
            self.clusters[agent_name].resample()

    def get_observation(
        self,
    ):
        obs = {}
        for name,cluster in self.clusters.items():
            obs[name] = {}
            pos, confidence = cluster.estimate_mean_position()
            obs[name]['pos'] = pos
            obs[name]['confidence'] = confidence

        return obs

    def get_positions_and_velocities(self):

        state_dict = {}
        for name, cluster in self.clusters.items():
            positions, velocities = cluster.get_state()
            state_dict[name] = (positions, velocities)
        return state_dict

class DronesParticleFilter(PredatorPreyParticleFilter):
    def __init__(
        self, 
        obs_map,
        agent_name,
        agent_start_pos: dict,  # e.g., {'agent0': [x0, y0], 'agent1': [x1, y1]}
        target_start_pos: list[float], 
        agent_control_function,
        target_control_function,
        num_particles: int = 100,
        resample_threshold: float = 3.0,
        std_dev: float = 0.2,
        max_speed: float = 3.0,
        speed_ratio: float = 0.8,
        dt: float = 0.333
    ):
        super().__init__(
            obs_map=obs_map,
            agent_name=agent_name,
            agent_start_pos=agent_start_pos,
            target_start_pos=target_start_pos,
            agent_control_function=agent_control_function,
            target_control_function=target_control_function,
            num_particles=num_particles,
            resample_threshold=resample_threshold,
            std_dev=std_dev,
            max_speed=max_speed,
            speed_ratio=speed_ratio,
            dt=dt
        )



class FootballParticleFilter(PredatorPreyParticleFilter):
    def __init__(
        self,
        obs_map,
        agent_name,
        agent_start_pos: dict,
        target_start_pos: list[float],
        agent_control_function,
        target_control_function,
        num_particles: int = 100,
        resample_threshold: float = 3.0,
        std_dev: float = 0.4,
        max_speed: float = 1.0,
        speed_ratio: float = 0.5,
        ball_speed: float = 0.1,
        measurement_std: float = 0.02,
        dt: float = 0.1
    ):
        self.obs_map = obs_map
        self.agent_name = agent_name
        self.n_agents = len(agent_start_pos.keys())
        self.target_pos = np.array(target_start_pos, dtype=float)
        self.clusters = {}
        self.dim = len(target_start_pos)
        self.dt = dt
        self.max_speed = max_speed
        self.speed_ratio = speed_ratio
        self.std_dev = std_dev

        for name, pos in agent_start_pos.items():
            self.clusters[name] = FootballParticleCluster(
                agent_control_function,
                num_particles=num_particles,
                mean_pos=np.array(pos, dtype=float),
                std_dev=np.ones(self.dim) * self.std_dev,
                resample_threshold=resample_threshold,
                dim=self.dim,
                max_speed=self.max_speed,
                dt=dt,
                target=False,
            )

        self.clusters['target'] = FootballParticleCluster(
            target_control_function,
            num_particles=num_particles,
            mean_pos=self.target_pos,
            std_dev=np.ones(self.dim) * self.std_dev * 3,
            resample_threshold=resample_threshold,
            dim=self.dim,
            max_speed=self.max_speed*self.speed_ratio,
            dt=dt,
            target=True,
        )

        self.agent_i = int(self.agent_name[-1])
        self.ball_speed = ball_speed
        self.measurement_std = measurement_std
        self.ball_position = None
        self.pass_start_pos = None
        self.pass_total_steps = None
        self.pass_steps = None
        self.receiver_idx = None
        self.last_action = None
        self.pass_prediction_stats = {'correct': 0, 'incorrect': 0}

    def reset(self, obs: dict):
        agent_names = [self.agent_name] + [name for name in self.clusters.keys() if 'agent' in name]
        ball_pos_global = obs[self.agent_name][self.obs_map['ball_pos']] + obs[self.agent_name][self.obs_map['self_pos']]
        closest_name = min(
            agent_names,
            key=lambda name: np.linalg.norm(obs[name][self.obs_map['self_pos']] - ball_pos_global),
        )

        self.ball_owner = np.zeros_like(obs[self.agent_name][self.obs_map['ball_owner']])
        self.ball_owner[int(closest_name[-1])] = 1
        self.ball_passing = False

        for name in self.clusters.keys():
            mean_pos = np.array(obs[name][self.obs_map['self_pos']], dtype=float)
            self.clusters[name].initialize_gaussian(mean_pos,np.ones(self.dim) * self.std_dev)

        self.init_obs = deepcopy(obs)

    def update_observation(
        self,
        agent_name: str,
        observed_pos: np.ndarray,
        action = None,
        measurement_std: float = None,
    ):

        if agent_name not in self.clusters:
            raise ValueError(f"Agent name '{agent_name}' not found in clusters.")

        if measurement_std is None:
            measurement_std = self.measurement_std

        force_reset = self.clusters[agent_name].update_weights(observed_pos, measurement_std)
        if not force_reset:
            self.clusters[agent_name].resample()

        if 'agent' in agent_name and self.ball_owner[int(agent_name[-1])] > 0:
            self.ball_position = deepcopy(observed_pos)
            self.ball_owner = np.zeros_like(self.ball_owner)
            self.ball_owner[int(agent_name[-1])] = 1
            if self.ball_passing:
                self.ball_passing = False
                self.pass_steps = None

        self.last_action = action

    def propagate_all(self, current_obs):

        if self.pass_steps is not None:
            self.pass_steps -= 1
            if self.pass_steps <= 0:
                true_ball_owner = current_obs[self.obs_map['ball_owner']]
                true_idx = int(np.argmax(true_ball_owner)) if true_ball_owner.max() > 0 else -1
                if true_idx == self.receiver_idx:
                    self.pass_prediction_stats['correct'] += 1
                else:
                    self.pass_prediction_stats['incorrect'] += 1

                self.ball_owner = np.zeros_like(self.ball_owner)
                self.ball_owner[self.receiver_idx] = 1
                self.ball_passing = False
                self.pass_steps = None

        if current_obs[self.obs_map['ball_owner']][self.agent_i] > 0:
            self.ball_owner = deepcopy(current_obs[self.obs_map['ball_owner']])

        team_positions = []
        for name in self.clusters.keys():
            mean_pos, _ = self.clusters[name].estimate_mean_position()
            if 'agent' in name:
                team_positions.append(mean_pos)
            elif name == 'target':
                target = {}
                target['pos'] = mean_pos
                target['vel'] = self.clusters[name].estimate_direction()

        team_positions = np.array(team_positions)
        team_ordered = np.insert(team_positions,self.agent_i,current_obs[self.obs_map['self_pos']],axis=0)

        PASS_ACTION = 9
        if self.last_action == PASS_ACTION and self.ball_owner[self.agent_i] > 0:
            self._begin_pass(team_ordered[self.agent_i], current_obs[self.obs_map['self_vel']], team_ordered, self.agent_i)
            self.last_action = None

        if self.ball_passing and self.pass_steps is not None:
            elapsed = self.pass_total_steps - self.pass_steps
            frac = np.clip(elapsed / self.pass_total_steps, 0.0, 1.0)
            receiver_pos = team_ordered[self.receiver_idx]
            self.ball_position = self.pass_start_pos + (receiver_pos - self.pass_start_pos) * frac
        elif self.ball_owner.max() > 0:
            owner_idx = int(np.argmax(self.ball_owner))
            if owner_idx < len(team_ordered):
                self.ball_position = team_ordered[owner_idx]
            else:
                self.ball_position = target['pos']

        for name in self.clusters.keys():
            if 'agent' in name:
                idx = int(name[-1])
                team_reordered = np.roll(team_ordered,-idx,axis=0)
                ball_owner_reordered = np.roll(self.ball_owner,-idx,axis=0)
                current_obs[self.obs_map['self_anchor']] = self.init_obs[name][self.obs_map['self_anchor']]
                passed = self.clusters[name].propagate(self.dt, current_obs, target, team_reordered[1:].flatten(), ball_owner_reordered, self.ball_position, self.obs_map)

                if passed:
                    self._begin_pass(team_ordered[idx], self.clusters[name].estimate_direction(), team_ordered, idx)
            else:
                self.clusters[name].propagate(self.dt, np.concatenate((current_obs,np.zeros((self.dim,)))), target, team_ordered.flatten(), self.ball_owner, -self.ball_position, self.obs_map)

        current_obs[self.obs_map['self_anchor']] = self.init_obs[self.agent_name][self.obs_map['self_anchor']]

    def _begin_pass(self, passer_pos, passer_dir, team_ordered, passer_idx):
        passer_norm = np.linalg.norm(passer_dir)
        if passer_norm > 0:
            passer_dir = passer_dir / passer_norm

        teammate_idxs = [j for j in range(len(team_ordered)) if j != passer_idx]
        rel_vecs = team_ordered[teammate_idxs] - passer_pos
        angles = [abs(np.arctan2(np.cross(passer_dir, v), np.dot(passer_dir, v))) for v in rel_vecs]
        receiver_idx = teammate_idxs[int(np.argmin(angles))]

        self.ball_passing = True
        self.receiver_idx = receiver_idx
        self.pass_distance = np.linalg.norm(team_ordered[receiver_idx] - passer_pos)
        self.pass_steps = int(np.ceil(self.pass_distance / (self.ball_speed * self.dt)))
        self.pass_start_pos = passer_pos
        self.pass_total_steps = max(self.pass_steps, 1)
        self.ball_owner = np.zeros_like(self.ball_owner)


    def get_observation(
        self,
    ):
        obs = {}
        for name,cluster in self.clusters.items():
            obs[name] = {}
            pos, confidence = cluster.estimate_mean_position()
            obs[name]['pos'] = pos
            obs[name]['confidence'] = confidence
            if 'target' in name:
                vel = self.clusters[name].estimate_direction()
                obs[name]['vel'] = vel

        obs['ball_owner'] = self.ball_owner
        obs['ball_pos'] = self.ball_position

        return obs

class FireParticleFilter(PredatorPreyParticleFilter):
    GREEN, RED, BLACK, WHITE, BLUE = 0, 1, 2, 3, 4

    def __init__(
        self,
        obs_map,
        agent_name,
        agent_start_pos: dict,
        target_start_pos: list[float],
        agent_control_function,
        target_control_function,
        num_particles: int = 10,
        resample_threshold: float = 50.0,
        std_dev: float = 1.0,
        max_speed: float = 1.0,
        speed_ratio: float = 0.5,
        measurement_std: float = 0.3,
        dt: float = 1.0
    ):
        self.obs_map = obs_map
        self.agent_name = agent_name
        self.n_agents = len(agent_start_pos.keys())
        self.target_pos = np.array(target_start_pos, dtype=float)
        self.clusters = {}
        self.dim = len(target_start_pos)
        self.dt = dt
        self.max_speed = max_speed
        self.std_dev = std_dev
        self.measurement_std = measurement_std
        self.target_control_function = target_control_function

        for name, pos in agent_start_pos.items():
            self.clusters[name] = FireDroneParticleCluster(
                agent_control_function,
                num_particles=num_particles,
                mean_pos=np.array(pos, dtype=float),
                std_dev=np.ones(self.dim) * self.std_dev,
                resample_threshold=resample_threshold,
                dim=self.dim,
                max_speed=self.max_speed,
                dt=dt,
                target=False,
            )

        self.clusters['target'] = None

    def reset(
        self,
        obs: dict,
    ):
        self.clusters['target'] = obs['target']
        for name in self.clusters.keys():
            if name == 'target':
                continue
            mean_pos = np.array(obs[name]['self_pos'], dtype=float)
            self.clusters[name].initialize_gaussian(mean_pos,np.ones(self.dim) * self.std_dev)

        self.observation_range = (len(obs[self.agent_name]['fire']) - 1) // 2

        self.fire = obs['target']
        self.fire_confidence = 1.0

        self.extinguish_confidence_decay = ((self.observation_range * 2 + 1) ** 2 - 100 ) / ((self.observation_range * 2 + 1) ** 2)

    def update_observation(
        self,
        agent_name: str,
        observed_value,
        action = None,
        measurement_std: float = None,
    ):
        if agent_name == 'target':
            self.fire = deepcopy(observed_value)
            self.fire_confidence = 1.0
            return

        if agent_name not in self.clusters:
            raise ValueError(f"Agent name '{agent_name}' not found in clusters.")

        if measurement_std is None:
            measurement_std = self.measurement_std

        force_reset = self.clusters[agent_name].update_weights(observed_value, measurement_std)
        if not force_reset:
            self.clusters[agent_name].resample()

    def propagate_all(self, current_obs):

        # Get estimated mean positions
        team_pos = {}
        for name in self.clusters.keys():
            if name == 'target':
                continue
            mean_pos, _ = self.clusters[name].estimate_mean_position()
            team_pos[name] = mean_pos

        team_pos[self.agent_name] = current_obs['self_pos']

        self.neighboring = self._count_green_neighboring_red(self.fire)
        self.fire_confidence *= ((self.observation_range * 2 + 1) ** 2 - self.neighboring ) / ((self.observation_range * 2 + 1) ** 2)
        self.fire = self.target_control_function(self.fire)

        target = self.fire

        # Propagate predator clusters
        for name in self.clusters.keys():
            if name == 'target':
                continue
            extinguish = self.clusters[name].propagate(self.dt, target, team_pos, name)
            if extinguish:
                pos, _ = self.clusters[name].estimate_mean_position()
                self.fire, ext_pix = self._extinguish_fire(pos, self.fire)
                self.fire_confidence *= self.extinguish_confidence_decay

    def get_observation(
        self,
        pos,
    ):
        obs = {}
        for name,cluster in self.clusters.items():
            if name == 'target':
                continue
            obs[name] = {}
            mean_pos, confidence = cluster.estimate_mean_position()
            obs[name]['pos'] = mean_pos
            obs[name]['confidence'] = confidence

        obs['target'] = {'confidence': self.fire_confidence}
        obs[self.agent_name] = self._combine_obs(obs, pos, self.fire)

        return obs

    def _local_fire_window(self, fire_state, agent_pos):
        r, c = agent_pos
        radius = self.observation_range

        padded_fire = np.pad(
            fire_state,
            pad_width=radius,
            mode="constant",
            constant_values=self.WHITE,
        )

        rp = r + radius
        cp = c + radius

        window = padded_fire[
            rp - radius: rp + radius + 1,
            cp - radius: cp + radius + 1,
        ]

        return window.astype(np.float32)

    def _combine_obs(self, obs, pos, fire):
        window_size = self.observation_range * 2 + 1
        partial_obs = self._local_fire_window(fire, pos)

        team = []

        for name in self.clusters.keys():
            if name == self.agent_name or name == 'target':
                continue

            team_pos = obs[name]['pos'] - pos
            team.append(team_pos)
            rel_r = int(round(team_pos[0]))
            rel_c = int(round(team_pos[1]))

            if abs(rel_r) <= self.observation_range and abs(rel_c) <= self.observation_range:
                cross_r = rel_r + self.observation_range
                cross_c = rel_c + self.observation_range
                for rr, cc in [(cross_r, cross_c), (cross_r - 1, cross_c), (cross_r + 1, cross_c), (cross_r, cross_c - 1), (cross_r, cross_c + 1)]:
                    if 0 <= rr < window_size and 0 <= cc < window_size:
                        partial_obs[rr, cc] = self.BLUE
        self.team = np.array(team).flatten()
        return partial_obs[..., None]

    def _extinguish_fire(
        self,
        pos,
        fire,
        max_extinguish_prob=1.0,
        decay_rate=0.05,
        extinguish_radius=5,
    ):
        """
        Extinguish nearby fire cells probabilistically.

        Probability decreases with distance from the drone:
            p = max_extinguish_prob * exp(-decay_rate * distance)

        Cells changed:
            RED -> BLACK with probability p
            GREEN stays GREEN
            BLACK stays BLACK
        """

        r, c = int(round(pos[0])), int(round(pos[1]))

        rows, cols = fire.shape

        r_min = max(0, r - extinguish_radius)
        r_max = min(rows, r + extinguish_radius + 1)
        c_min = max(0, c - extinguish_radius)
        c_max = min(cols, c + extinguish_radius + 1)

        rr, cc = np.meshgrid(
            np.arange(r_min, r_max),
            np.arange(c_min, c_max),
            indexing="ij",
        )
        distance = np.sqrt((rr - r) ** 2 + (cc - c) ** 2)

        window = fire[r_min:r_max, c_min:c_max]
        in_range_red = (window == self.RED) & (distance <= extinguish_radius)

        extinguish_prob = max_extinguish_prob * np.exp(-decay_rate * distance)
        trial = np.random.random(distance.shape) < extinguish_prob

        newly_extinguished = in_range_red & trial
        window[newly_extinguished] = self.BLACK

        return fire, int(np.sum(newly_extinguished))

    def _count_green_neighboring_red(self, fire):

        red_mask = fire == self.RED
        green_mask = fire == self.GREEN
        rows, cols = fire.shape

        padded_red = np.pad(red_mask, 1, mode="constant", constant_values=False)
        neighbor_offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]

        adjacent_to_red = np.zeros_like(green_mask)
        for dr, dc in neighbor_offsets:
            adjacent_to_red |= padded_red[1 + dr: 1 + dr + rows, 1 + dc: 1 + dc + cols]

        return int(np.sum(green_mask & adjacent_to_red))