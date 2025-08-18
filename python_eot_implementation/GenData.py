"""
Data Generation for Extended Object Tracking (EOT) Simulation

This module contains the Plant class responsible for generating ground truth
target trajectories and simulated measurements for the EOT algorithm.
"""

import numpy as np
from typing import List, Tuple
from scipy.stats import invwishart, poisson
from dataclasses import dataclass


@dataclass
class EOTParameters:
    """
    Data class containing all parameters for the EOT algorithm.
    
    This structure organizes the many parameters needed for the tracking algorithm
    into logical groups for better maintainability.
    """
    
    # Temporal parameters
    scan_time: float = 0.2
    
    # Target dynamics parameters  
    acceleration_deviation: float = 1.0
    survival_probability: float = 0.99
    mean_births: float = 0.01
    
    # Measurement parameters
    surveillance_region: np.ndarray = None
    measurement_variance: float = 1.0
    mean_measurements: float = 8.0
    mean_clutter: float = 10.0
    
    # Prior distribution parameters
    prior_velocity_covariance: np.ndarray = None
    prior_extent_1: np.ndarray = None
    prior_extent_2: float = 100.0
    degree_freedom_prediction: float = 20000.0
    
    # Sampling parameters
    num_particles: int = 5000
    regularization_deviation: float = 0.0
    
    # Detection and pruning parameters
    detection_threshold: float = 0.5
    threshold_pruning: float = 1e-3
    minimum_track_length: int = 1
    
    # Message passing parameters
    num_outer_iterations: int = 2


class Plant:
    """
    Plant class for generating ground truth data and measurements for EOT simulation.
    
    This class handles:
    - Generation of initial target states and extents
    - Simulation of target trajectories using motion models
    - Creation of noisy measurements with clutter
    - Management of target appearance/disappearance
    """
    
    def __init__(self, parameters: EOTParameters):
        """
        Initialize the Plant with simulation parameters.
        
        Args:
            parameters: EOT parameters containing simulation settings
        """
        self.parameters = parameters
        
    def get_transition_matrices(self, scan_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate state transition matrices for constant velocity motion model.
        
        This implements a 2D constant velocity motion model with acceleration noise.
        The state vector is [x, y, vx, vy] where (x,y) is position and (vx,vy) is velocity.
        
        Args:
            scan_time: Time interval between measurements
            
        Returns:
            A: State transition matrix (4x4)
            W: Process noise input matrix (4x2)
            
        Mathematical model:
            x(k+1) = A * x(k) + W * w(k)
            where w(k) is 2D acceleration noise
        """
        # State transition matrix for [x, y, vx, vy]
        A = np.eye(4)
        A[0, 2] = scan_time  # x = x + vx * dt
        A[1, 3] = scan_time  # y = y + vy * dt
        
        # Process noise input matrix (maps 2D acceleration to 4D state)
        W = np.zeros((4, 2))
        W[0, 0] = 0.5 * scan_time**2  # x component from x-acceleration
        W[1, 1] = 0.5 * scan_time**2  # y component from y-acceleration  
        W[2, 0] = scan_time           # vx component from x-acceleration
        W[3, 1] = scan_time           # vy component from y-acceleration
        
        return A, W

    def get_start_states(self, num_targets: int, radius: float, speed: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate initial target states arranged in a circle.
        
        This function creates initial conditions for multiple targets arranged
        in a circular formation, each moving tangentially to the circle.
        
        Args:
            num_targets: Number of targets to initialize
            radius: Radius of circular formation
            speed: Initial speed of targets
            
        Returns:
            start_states: Initial kinematic states (4 x num_targets)
            start_matrices: Initial extent matrices (2 x 2 x num_targets)
        """
        # Generate initial extent matrices from inverse Wishart distribution
        start_matrices = np.zeros((2, 2, num_targets))
        for target in range(num_targets):
            # Sample extent matrix from inverse Wishart prior
            start_matrices[:, :, target] = invwishart.rvs(
                df=self.parameters.prior_extent_2,
                scale=self.parameters.prior_extent_1
            )
        
        # Special case for single target
        if num_targets < 2:
            start_states = np.array([[0], [radius], [0], [-speed]])
            return start_states, start_matrices
        
        # Initialize state array: [x, y, vx, vy] for each target
        start_states = np.zeros((4, num_targets))
        
        # First target starts at (0, radius) moving left
        start_states[:, 0] = [0, radius, 0, -speed]
        
        # Arrange remaining targets in circle with equal angular spacing
        step_size = 2 * np.pi / num_targets
        
        for target in range(1, num_targets):
            angle = target * step_size
            # Position on circle
            x = np.sin(angle) * radius
            y = np.cos(angle) * radius
            # Velocity tangent to circle (perpendicular to radius)
            vx = -np.sin(angle) * speed
            vy = -np.cos(angle) * speed
            
            start_states[:, target] = [x, y, vx, vy]
        
        return start_states, start_matrices

    def generate_tracks_unknown(self, start_states: np.ndarray, extent_matrices: np.ndarray, 
                               appearance_from_to: np.ndarray, num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ground truth target trajectories with appearance/disappearance times.
        
        This function simulates target motion using a constant velocity model with
        acceleration noise. Targets can appear and disappear at specified times.
        
        Args:
            start_states: Initial target states (4 x num_targets)
            extent_matrices: Target extent matrices (2 x 2 x num_targets)
            appearance_from_to: Appearance/disappearance times (2 x num_targets)
            num_steps: Number of simulation time steps
            
        Returns:
            target_tracks: Target trajectories (4 x num_steps x num_targets)
            target_extents: Target extent matrices (2 x 2 x num_steps x num_targets)
        """
        acceleration_deviation = self.parameters.acceleration_deviation
        scan_time = self.parameters.scan_time
        num_targets = start_states.shape[1]
        
        # Get motion model matrices
        A, W = self.get_transition_matrices(scan_time)
        
        # Initialize output arrays (NaN indicates target not present)
        target_tracks = np.full((4, num_steps, num_targets), np.nan)
        target_extents = np.full((2, 2, num_steps, num_targets), np.nan)
        
        # Simulate each target trajectory
        for target in range(num_targets):
            # Current state starts from initial condition
            current_state = start_states[:, target].copy()
            
            # Get appearance/disappearance times (convert to 0-indexed)
            appear_time = int(appearance_from_to[0, target] - 1)
            disappear_time = int(appearance_from_to[1, target] - 1)
            
            # Simulate trajectory for all time steps
            for step in range(num_steps):
                # Apply motion model with acceleration noise
                noise = W @ (acceleration_deviation * np.random.randn(2))
                current_state = A @ current_state + noise
                
                # Store state only if target is supposed to be visible
                if appear_time <= step <= disappear_time:
                    target_tracks[:, step, target] = current_state
                    target_extents[:, :, step, target] = extent_matrices[:, :, target]
        
        return target_tracks, target_extents

    def generate_cluttered_measurements(self, target_tracks: np.ndarray, 
                                       target_extents: np.ndarray) -> List[np.ndarray]:
        """
        Generate noisy measurements from targets plus clutter (false alarms).
        
        This function simulates the measurement process:
        1. For each visible target, generate Poisson-distributed number of measurements
        2. Each measurement is drawn from target position + extent noise + sensor noise
        3. Add uniformly distributed clutter measurements
        4. Randomly permute all measurements
        
        Args:
            target_tracks: Ground truth target trajectories
            target_extents: Ground truth target extents  
            
        Returns:
            cluttered_measurements: List of measurement arrays, one per time step
        """
        mean_clutter = self.parameters.mean_clutter
        measurement_variance = self.parameters.measurement_variance
        mean_measurements = self.parameters.mean_measurements
        surveillance_region = self.parameters.surveillance_region
        
        num_steps = target_tracks.shape[1]
        num_targets = target_tracks.shape[2]
        
        cluttered_measurements = []
        
        for step in range(num_steps):
            measurements = np.empty((2, 0))  # Start with empty measurements
            
            # Generate measurements from each visible target
            for target in range(num_targets):
                # Skip if target not present at this time
                if np.isnan(target_tracks[0, step, target]):
                    continue
                    
                # Get target position and extent
                target_pos = target_tracks[:2, step, target]
                target_extent = target_extents[:, :, step, target]
                
                # Number of measurements from this target (Poisson distributed)
                num_meas = np.random.poisson(mean_measurements)
                
                if num_meas > 0:
                    # Generate measurements from target extent
                    # Each measurement = target_position + extent_noise + sensor_noise
                    extent_noise = np.random.multivariate_normal(
                        mean=np.zeros(2),
                        cov=target_extent @ target_extent,  # extent^2 as covariance
                        size=num_meas
                    ).T
                    
                    sensor_noise = np.sqrt(measurement_variance) * np.random.randn(2, num_meas)
                    
                    target_measurements = (target_pos.reshape(-1, 1) + 
                                         extent_noise + sensor_noise)
                    
                    # Concatenate to measurement list
                    measurements = np.concatenate([measurements, target_measurements], axis=1)
            
            # Generate false alarm (clutter) measurements
            num_false_alarms = np.random.poisson(mean_clutter)
            
            if num_false_alarms > 0:
                # Uniform distribution over surveillance region
                false_alarms = np.random.uniform(
                    low=[surveillance_region[0, 0], surveillance_region[1, 0]],
                    high=[surveillance_region[0, 1], surveillance_region[1, 1]],
                    size=(num_false_alarms, 2)
                ).T
                
                # Add false alarms to measurements
                measurements = np.concatenate([false_alarms, measurements], axis=1)
            
            # Randomly shuffle measurements (target and clutter mixed)
            if measurements.shape[1] > 0:
                perm = np.random.permutation(measurements.shape[1])
                measurements = measurements[:, perm]
            
            cluttered_measurements.append(measurements)
        
        return cluttered_measurements

    def generate_scenario(self, num_targets: int, num_steps: int, start_radius: float = 75, 
                         start_velocity: float = 10, appearance_from_to: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Generate a complete simulation scenario with ground truth and measurements.
        
        This is a convenience method that generates initial states, simulates trajectories,
        and creates measurements in one call.
        
        Args:
            num_targets: Number of targets to simulate
            num_steps: Number of time steps
            start_radius: Radius of circular formation
            start_velocity: Initial velocity magnitude
            appearance_from_to: Target appearance/disappearance times (2 x num_targets)
            
        Returns:
            target_tracks: Ground truth trajectories
            target_extents: Ground truth extents
            measurements: Simulated measurements with clutter
        """
        # Generate initial states
        start_states, start_matrices = self.get_start_states(num_targets, start_radius, start_velocity)
        
        # Default appearance times if not specified
        if appearance_from_to is None:
            appearance_from_to = np.array([
                [3, 83], [3, 83], [6, 86], [6, 86], [9, 89],
                [9, 89], [12, 92], [12, 92], [15, 95], [15, 95]
            ]).T
        
        # Generate trajectories
        target_tracks, target_extents = self.generate_tracks_unknown(
            start_states, start_matrices, appearance_from_to, num_steps
        )
        
        # Generate measurements
        measurements = self.generate_cluttered_measurements(target_tracks, target_extents)
        
        return target_tracks, target_extents, measurements