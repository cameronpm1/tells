import time
import numpy as np
from typing import List
from copy import deepcopy
import matplotlib.pyplot as plt

import numpy as np

class Particle:
    def __init__(self, 
        position, 
        velocity_dir, 
        speed, 
        dim=2, 
        max_speed = 2,
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
        num_particles: int = 100,
        mean_pos: list[float] = [0.0,0.0],
        std_dev: list[float] = [0.0,0.0],
        dim: int = 2,
        max_speed: float = 0.4,
        dt: float = 0.1,
        prey = False,
    ):
        """
        Args:
            num_particles (int): number of particles in the cluster
        """
        self.num_particles = num_particles
        self.dim = dim
        self.max_speed = max_speed
        self.dt = dt
        self.prey = prey
        self.particles = []
        self.weights = np.ones(num_particles) / num_particles

        if self.prey:
            self.control_func = self.simple_prey_control
        else:
            self.control_func = self.simple_pred_control
        
        self.initialize_gaussian(mean_pos,std_dev)
    
    def initialize_gaussian(self, mean_pos, std_dev):

        del self.particles

        self.particles = []
        mean_pos = np.array(mean_pos)
        
        if isinstance(std_dev, (float, int)):
            std_x = std_y = float(std_dev)
        else:
            std_x, std_y = std_dev
        
        for _ in range(self.num_particles):
            x = np.random.normal(mean_pos[0], std_x)
            y = np.random.normal(mean_pos[1], std_y)
            position = np.array([x, y])
            
            # Random unit velocity direction
            angle = np.random.uniform(0, 2*np.pi)
            velocity_dir = np.array([np.cos(angle), np.sin(angle)])
            
            speed = 0.0  # all speeds set to zero
            
            self.particles.append(Particle(position, velocity_dir, speed, self.dim, self.max_speed))
        
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def propagate(self, dt, prey_pos, predator_positions, goal_pos):

        num_particles = len(self.particles)
        
        for i, particle in enumerate(self.particles):
            if not self.prey:
                predator_positions = np.vstack([
                    predator_positions,
                    particle.position,
                ])
            else:
                prey_pos = particle.position

            vel_cmd = self.control_func(predator_positions, prey_pos, goal_pos)

            particle.add_control(vel_cmd)
            particle.propagate(dt)
    
    def resample(self):
        """
        Resample particles according to their weights using systematic resampling.
        """
        '''
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # avoid round-off error
        step = 1.0 / self.num_particles
        start = np.random.uniform(0, step)
        positions = start + step * np.arange(self.num_particles)
        
        indexes = np.zeros(self.num_particles, dtype=int)
        i, j = 0, 0
        while i < self.num_particles:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        
        # Resample particles and reset weights
        self.particles = [self.particles[idx] for idx in indexes]
        self.weights.fill(1.0 / self.num_particles)
        '''

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
        return mean_pos, np.average(np.std(positions,axis=0)) #np.average(np.stdev(positions, axis=0))
    
    def update_weights(self, measurement, measurement_std):
        """
        Update particle weights based on a measurement (e.g., observed position).

        Args:
            measurement (np.array): observed 2D position
            measurement_std (float): standard deviation of measurement noise
        """

        positions = np.array([p.position for p in self.particles])
        dist = np.linalg.norm(positions - measurement, axis=1)
        if min(dist) > 3.0:
            self.initialize_gaussian(measurement,np.ones(measurement.shape)*measurement_std)
            return True
        likelihoods = np.exp(-0.5 * dist**2) # / (measurement_std**2))
        self.weights = likelihoods + 1e-300  # avoid zeros
        #print(max(self.weights),min(self.weights),np.average(self.weights),np.std(self.weights))
        self.weights /= np.sum(self.weights)
        return False
        



    def get_state(self):

        positions = np.array([p.position for p in self.particles])
        velocities = np.array([p.velocity_dir * p.speed for p in self.particles])
        return positions, velocities

    def simple_pred_control(self, pred_pos, prey_pos, goal_pos):

        forward = goal_pos - prey_pos
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        lateral = np.array([-forward[1], forward[0]])
        pos = pred_pos[-1]

        slots = []
        backoff = 3.0
        lateral_spacing = 2.0
        num_slots = len(pred_pos)

        for i in range(num_slots):
            offset = (i - (num_slots - 1) / 2) * lateral_spacing
            slot = prey_pos - (forward * backoff) + (lateral * offset)
            slots.append(slot)
        
        closest_slot = min(slots, key=lambda s: np.linalg.norm(pos - s))

        #compute control for last predator
        pos = pred_pos[-1]
        vec = closest_slot - pos
        dist = np.linalg.norm(vec)
        command = vec / dist if dist > 0.1 else np.zeros(2)

        return command

    def simple_prey_control(self, predator_positions, prey_pos, _):
        """
        Compute prey velocity vector repelled by predators with fixed parameters.

        Args:
            prey_pos (np.ndarray): 2D position of the prey.
            predator_positions (np.ndarray): Array of shape (N, 2) with predator positions.

        Returns:
            np.ndarray: Normalized 2D velocity vector for the prey.
        """
        prey_sensitivity = 1.2
        prey_avoid_radius = 2.4
        prey_avoid_gain = 1.5
        force_exponent = 2.5

        force = np.zeros(2)

        for predator_pos in predator_positions:
            diff = prey_pos - predator_pos
            dist = np.linalg.norm(diff) + 1e-6  # avoid division by zero
            gain = prey_sensitivity
            if dist < prey_avoid_radius:
                gain *= prey_avoid_gain
            force += gain * (diff / dist ** force_exponent)
        
        norm = np.linalg.norm(force)
        if norm > 0:
            force /= norm

        return force

class PredatorPreyParticleFilter:
    def __init__(
        self, 
        agent_start_pos: dict,  # e.g., {'agent0': [x0, y0], 'agent1': [x1, y1]}
        prey_start_pos: list[float], 
        num_particles: int = 100,
        std_dev: float = 0.2,
        max_speed: float = 1.0,
        dt: float = 0.1
    ):
        self.n_agents = len(agent_start_pos.keys())
        self.prey_pos = np.array(prey_start_pos, dtype=float)
        self.clusters = {}
        self.dim = len(prey_start_pos)
        self.dt = dt
        self.max_speed = max_speed
        self.std_dev = std_dev

        for i, (name, pos) in enumerate(agent_start_pos.items()):
            mean_pos = np.array(pos, dtype=float)
            cluster = ParticleCluster(
                num_particles=num_particles,
                mean_pos=mean_pos,
                std_dev=np.ones(self.dim) * self.std_dev,
                dim=self.dim,
                max_speed=self.max_speed,
                dt=dt,
                prey=False,
            )
            self.clusters[name] = cluster

        # Initialize prey cluster
        self.clusters['target'] = ParticleCluster(
            num_particles=num_particles,
            mean_pos=self.prey_pos,
            std_dev=np.ones(self.dim) * self.std_dev,
            dim=self.dim,
            max_speed=self.max_speed*.4,
            dt=dt,
            prey=True,
        )


    def reset(
        self,
        agent_start_pos: dict,  # e.g., {'agent0': [x0, y0], 'agent1': [x1, y1]}
        prey_start_pos: list[float], 
    ):

        # Initialize one ParticleCluster per predator
        for i, (name, pos) in enumerate(agent_start_pos.items()):
            mean_pos = np.array(pos, dtype=float)
            self.clusters[name].initialize_gaussian(mean_pos,np.ones(self.dim) * self.std_dev)

        # Initialize prey cluster
        self.clusters['target'].initialize_gaussian(prey_start_pos,np.ones(self.dim) * self.std_dev)

    def propagate_all(self, current_pos, goal_pos):

        predator_positions = []
        prey_position = None

        # Get estimated mean positions
        for name in self.clusters.keys():
            mean_pos, _ = self.clusters[name].estimate_mean_position()
            if 'agent' in name:
                predator_positions.append(mean_pos)
            elif name == 'target':
                prey_position = mean_pos

        predator_positions = np.array(predator_positions)

        # Propagate predator clusters
        for i, name in enumerate(self.clusters.keys()):
            if 'agent' in name:
                # Other predators excluding current one
                other_preds = np.delete(deepcopy(predator_positions), i, axis=0)
                self.clusters[name].propagate(self.dt, prey_position, other_preds, goal_pos)
            elif name == 'target':
                # For prey, pass all predator positions (estimated + current true)
                all_preds = np.vstack([predator_positions, current_pos])
                self.clusters[name].propagate(self.dt, prey_position, all_preds, goal_pos)

    def update_observation(
        self, 
        agent_name: str, 
        observed_pos: np.ndarray, 
        measurement_std: float = 0.1
    ):

        if agent_name not in self.clusters:
            raise ValueError(f"Agent name '{agent_name}' not found in clusters.")
        #est,_ = self.clusters[agent_name].estimate_mean_position()
        #print(est,observed_pos)
        force_reset = self.clusters[agent_name].update_weights(observed_pos, measurement_std)
        if not force_reset:
            self.clusters[agent_name].resample()
        #est,_ = self.clusters[agent_name].estimate_mean_position()
        #print(est)

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


def test_two_predator_env():
    # Initial positions
    agent_start_pos = {
        'agent0': [2.0, 3.0],
        'agent1': [4.0, 5.0]
    }
    prey_start_pos = [5.0, 5.0]
    goal_pos = np.array([10.0, 10.0])

    # Initialize tracker
    tracker = PredatorPreyParticleFilter(agent_start_pos, prey_start_pos, num_particles=100, std_dev=0.3)

    dt = 0.1
    num_steps = 300

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    for step in range(num_steps):
        ax.clear()

        # Get current estimated predator positions
        predator_positions = []
        for name in ['agent0', 'agent1']:
            pos, _ = tracker.clusters[name].estimate_mean_position()
            predator_positions.append(pos)
        predator_positions = np.array(predator_positions)

        # Propagate all clusters using current predator positions
        tracker.propagate_all(current_pos=predator_positions, goal_pos=goal_pos)

        # Get prey estimated position
        prey_pos, _ = tracker.clusters['target'].estimate_mean_position()

        # Plotting
        ax.scatter(goal_pos[0], goal_pos[1], c='green', label='Goal')
        ax.scatter(prey_pos[0], prey_pos[1], c='red', label='Target')

        # Plot predators
        for name in ['agent0', 'agent1']:
            positions, velocities = tracker.clusters[name].get_state()
            ax.scatter(positions[:, 0], positions[:, 1], alpha=0.5, label=f'{name} particles')
            scale = 0.3
            for pos, vel in zip(positions, velocities):
                ax.arrow(pos[0], pos[1], vel[0]*scale, vel[1]*scale,
                         head_width=0.05, head_length=0.05, fc='blue', ec='blue', alpha=0.5)

        # Plot prey particles
        positions, velocities = tracker.clusters['target'].get_state()
        ax.scatter(positions[:, 0], positions[:, 1], alpha=0.5, c='orange', label='Prey particles')
        scale = 0.3
        for pos, vel in zip(positions, velocities):
            ax.arrow(pos[0], pos[1], vel[0]*scale, vel[1]*scale,
                     head_width=0.05, head_length=0.05, fc='orange', ec='orange', alpha=0.5)

        ax.set_xlim(0, 15)
        ax.set_ylim(0, 15)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
        ax.legend(loc='upper left')
        ax.set_title(f"Step {step+1}")
        plt.pause(0.1)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    test_two_predator_env()
