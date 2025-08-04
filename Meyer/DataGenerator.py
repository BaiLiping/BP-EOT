import numpy as np
from scipy.stats import wishart, poisson
from typing import Tuple, List, Dict, Any

class DataGenerator:
    """
    Python implementation of data generation for Extended Object Tracking simulation.
    Converts MATLAB functions: generateTracksUnknown.m, generateClutteredMeasurements.m, 
    getStartStates.m, and getTransitionMatrices.m
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize data generator with simulation parameters.
        
        Args:
            parameters: Dictionary containing simulation parameters matching MATLAB main.m
        """
        self.parameters = parameters
        
    def get_transition_matrices(self, scan_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate state transition matrices for constant velocity model.
        Port of getTransitionMatrices.m
        
        Args:
            scan_time: Time step between scans
            
        Returns:
            A: State transition matrix (4x4)
            W: Process noise mapping matrix (4x2)  
            reducedA: Reduced transition matrix (2x2)
            reducedW: Reduced noise mapping (2x1)
        """
        A = np.eye(4)
        A[0, 2] = scan_time  # position-velocity coupling in x
        A[1, 3] = scan_time  # position-velocity coupling in y
        
        W = np.zeros((4, 2))
        W[0, 0] = 0.5 * scan_time**2  # acceleration effect on x position
        W[1, 1] = 0.5 * scan_time**2  # acceleration effect on y position  
        W[2, 0] = scan_time           # acceleration effect on x velocity
        W[3, 1] = scan_time           # acceleration effect on y velocity
        
        # Reduced matrices for 2D position-only dynamics
        reducedA = A[[0, 2], :][:, [0, 2]]  # x position and velocity only
        reducedW = W[[0, 2], 0]             # corresponding process noise
        
        return A, W, reducedA, reducedW
        
    def get_start_states(self, num_targets: int, radius: float, 
                        speed: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate initial target states and extent matrices.
        Port of getStartStates.m
        
        Args:
            num_targets: Number of targets to initialize
            radius: Initial radius for circular target placement
            speed: Initial speed magnitude
            
        Returns:
            start_states: Initial kinematic states (4 x num_targets)
            start_matrices: Initial extent matrices (2 x 2 x num_targets)
        """
        prior_extent1 = self.parameters['priorExtent1']
        prior_extent2 = self.parameters['priorExtent2']
        
        # Sample initial extent matrices using inverse Wishart distribution
        start_matrices = np.zeros((2, 2, num_targets))
        for target in range(num_targets):
            start_matrices[:, :, target] = wishart.rvs(
                df=prior_extent2, 
                scale=np.linalg.inv(prior_extent1)
            )
            start_matrices[:, :, target] = np.linalg.inv(start_matrices[:, :, target])
            
        # Initialize kinematic states
        if num_targets < 2:
            start_states = np.zeros((4, 1))
            start_states[2, 0] = speed  # x velocity
        else:
            start_states = np.zeros((4, num_targets))
            start_states[:, 0] = [0, radius, 0, -speed]  # First target
            
            step_size = 2 * np.pi / num_targets
            
            for target in range(1, num_targets):
                angle = target * step_size
                start_states[:, target] = [
                    np.sin(angle) * radius,    # x position
                    np.cos(angle) * radius,    # y position  
                    -np.sin(angle) * speed,    # x velocity
                    -np.cos(angle) * speed     # y velocity
                ]
                
        return start_states, start_matrices
        
    def generate_tracks_unknown(self, start_states: np.ndarray, 
                               extent_matrices: np.ndarray,
                               appearance_from_to: np.ndarray, 
                               num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate true target tracks with time-varying appearance/disappearance.
        Port of generateTracksUnknown.m
        
        Args:
            start_states: Initial states (4 x num_targets)
            extent_matrices: Initial extent matrices (2 x 2 x num_targets)  
            appearance_from_to: Appearance intervals (2 x num_targets)
            num_steps: Number of time steps
            
        Returns:
            target_tracks: True target tracks (4 x num_steps x num_targets)
            target_extents: True target extents (2 x 2 x num_steps x num_targets)
        """
        acceleration_deviation = self.parameters['accelerationDeviation']
        scan_time = self.parameters['scanTime']
        num_targets = start_states.shape[1]
        
        # Get transition matrices
        A, W, _, _ = self.get_transition_matrices(scan_time)
        
        # Initialize output arrays
        target_tracks = np.full((4, num_steps, num_targets), np.nan)
        target_extents = np.full((2, 2, num_steps, num_targets), np.nan)
        
        # Generate tracks for each target
        for target in range(num_targets):
            current_state = start_states[:, target].copy()
            
            for step in range(num_steps):
                # Propagate state with process noise
                process_noise = W @ (acceleration_deviation * np.random.randn(2))
                current_state = A @ current_state + process_noise
                
                # Check if target is visible at this time step
                step_idx = step + 1  # MATLAB uses 1-based indexing
                if (step_idx >= appearance_from_to[0, target] and 
                    step_idx <= appearance_from_to[1, target]):
                    target_tracks[:, step, target] = current_state
                    target_extents[:, :, step, target] = extent_matrices[:, :, target]
                    
        return target_tracks, target_extents
        
    def generate_cluttered_measurements(self, target_tracks: np.ndarray, 
                                      target_extents: np.ndarray) -> List[np.ndarray]:
        """
        Generate noisy measurements with clutter for all time steps.
        Port of generateClutteredMeasurements.m
        
        Args:
            target_tracks: True target tracks (4 x num_steps x num_targets)
            target_extents: True target extents (2 x 2 x num_steps x num_targets)
            
        Returns:
            cluttered_measurements: List of measurement arrays for each time step
        """
        mean_clutter = self.parameters['meanClutter']
        measurement_variance = self.parameters['measurementVariance']
        mean_measurements = self.parameters['meanMeasurements']
        surveillance_region = self.parameters['surveillanceRegion']
        
        num_steps = target_tracks.shape[1]
        num_targets = target_tracks.shape[2]
        
        cluttered_measurements = []
        
        for step in range(num_steps):
            measurements = np.empty((2, 0))
            
            # Generate measurements from each visible target
            for target in range(num_targets):
                if np.isnan(target_tracks[0, step, target]):
                    continue
                    
                # Number of measurements from this target (Poisson distributed)
                num_measurements_tmp = poisson.rvs(mean_measurements)
                
                if num_measurements_tmp > 0:
                    # Target position
                    target_pos = target_tracks[:2, step, target]
                    
                    # Sample from target extent (multivariate normal)
                    extent_squared = np.linalg.matrix_power(target_extents[:, :, step, target], 2)
                    extent_samples = np.random.multivariate_normal(
                        np.zeros(2), extent_squared, num_measurements_tmp
                    ).T
                    
                    # Add target position and measurement noise
                    measurements_tmp = (target_pos[:, np.newaxis] + extent_samples + 
                                      np.sqrt(measurement_variance) * np.random.randn(2, num_measurements_tmp))
                    
                    measurements = np.concatenate([measurements, measurements_tmp], axis=1)
            
            # Generate false alarms (clutter)
            num_false_alarms = poisson.rvs(mean_clutter)
            
            if num_false_alarms > 0:
                false_alarms = np.zeros((2, num_false_alarms))
                # Uniform distribution over surveillance region
                false_alarms[0, :] = (surveillance_region[1, 0] - surveillance_region[0, 0]) * \
                                   np.random.rand(num_false_alarms) + surveillance_region[0, 0]
                false_alarms[1, :] = (surveillance_region[1, 1] - surveillance_region[0, 1]) * \
                                   np.random.rand(num_false_alarms) + surveillance_region[0, 1]
                
                measurements = np.concatenate([false_alarms, measurements], axis=1)
            
            # Randomly permute measurements to mix true detections and clutter
            if measurements.shape[1] > 0:
                perm_indices = np.random.permutation(measurements.shape[1])
                measurements = measurements[:, perm_indices]
                
            cluttered_measurements.append(measurements)
            
        return cluttered_measurements
        
    def simulate_scenario(self, num_steps: int, num_targets: int, 
                         mean_target_dimension: float, start_radius: float, 
                         start_velocity: float, 
                         appearance_from_to: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Complete simulation pipeline matching main.m
        
        Args:
            num_steps: Number of time steps
            num_targets: Number of targets
            mean_target_dimension: Mean target size
            start_radius: Initial placement radius
            start_velocity: Initial speed
            appearance_from_to: Target appearance intervals
            
        Returns:
            target_tracks: True target trajectories
            target_extents: True target extents
            measurements: Cluttered measurements for each time step
        """
        # Generate initial states
        start_states, start_matrices = self.get_start_states(num_targets, start_radius, start_velocity)
        
        # Generate true tracks
        target_tracks, target_extents = self.generate_tracks_unknown(
            start_states, start_matrices, appearance_from_to, num_steps
        )
        
        # Generate measurements with clutter
        measurements = self.generate_cluttered_measurements(target_tracks, target_extents)
        
        return target_tracks, target_extents, measurements