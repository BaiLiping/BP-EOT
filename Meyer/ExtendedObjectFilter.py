import numpy as np
from scipy.stats import wishart, multivariate_normal
from typing import List, Dict, Any, Tuple, Optional
import copy

class ExtendedObjectFilter:
    """
    Python implementation of Extended Object Tracking filter using belief propagation.
    Converts MATLAB function: eotEllipticalShape.m and related helper functions
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize Extended Object Filter with tracking parameters.
        
        Args:
            parameters: Dictionary containing tracking parameters from main.m
        """
        self.parameters = parameters
        
    def get_log_weights_fast(self, measurement: np.ndarray, 
                           particles_kinematic: np.ndarray,
                           particle_extents: np.ndarray) -> np.ndarray:
        """
        Compute log-likelihood weights for particles given measurement.
        Port of getLogWeightsFast.m
        
        Args:
            measurement: Current measurement (2,)
            particles_kinematic: Particle positions (2, num_particles)
            particle_extents: Particle covariance matrices (2, 2, num_particles)
            
        Returns:
            log_weights: Log-likelihood weights (num_particles,)
        """
        num_particles = particle_extents.shape[2]
        
        # Compute determinants for all particles
        det_all = (particle_extents[0, 0, :] * particle_extents[1, 1, :] - 
                   particle_extents[0, 1, :] ** 2)
        
        # Normalization factor
        log_factors = np.log(1.0 / (2 * np.pi * np.sqrt(det_all)))
        
        # Innovation (measurement - predicted position)
        innovation = measurement[:, np.newaxis] - particles_kinematic
        
        # Efficiently compute innovation^T * inv(covariance) * innovation
        inv_det = 1.0 / det_all
        
        # Matrix inverse using determinant formula for 2x2 matrices
        # Fix broadcasting: inv_det is (num_particles,), innovation.T is (num_particles, 2)
        temp = inv_det[:, np.newaxis] * innovation.T
        part1 = np.zeros((num_particles, 2))
        part1[:, 0] = temp[:, 0] * particle_extents[1, 1, :] - temp[:, 1] * particle_extents[1, 0, :]
        part1[:, 1] = -temp[:, 0] * particle_extents[0, 1, :] + temp[:, 1] * particle_extents[0, 0, :]
        
        # Quadratic form
        quadratic = -0.5 * (part1[:, 0] * innovation[0, :] + part1[:, 1] * innovation[1, :])
        
        log_weights = log_factors + quadratic
        
        return log_weights
        
    def get_square2_fast(self, matrices_in: np.ndarray) -> np.ndarray:
        """
        Compute matrix square operation efficiently.
        Port of getSquare2Fast.m
        
        Args:
            matrices_in: Input matrices (2, 2, num_particles)
            
        Returns:
            matrices_out: Squared matrices (2, 2, num_particles)
        """
        matrices_out = matrices_in.copy()
        
        matrices_out[0, 0, :] = (matrices_in[0, 0, :] * matrices_in[0, 0, :] + 
                                matrices_in[1, 0, :] * matrices_in[1, 0, :])
        matrices_out[1, 0, :] = (matrices_in[1, 0, :] * matrices_in[0, 0, :] + 
                                matrices_in[1, 1, :] * matrices_in[1, 0, :])
        matrices_out[0, 1, :] = (matrices_in[0, 0, :] * matrices_in[0, 1, :] + 
                                matrices_in[0, 1, :] * matrices_in[1, 1, :])
        matrices_out[1, 1, :] = (matrices_in[1, 0, :] * matrices_in[0, 1, :] + 
                                matrices_in[1, 1, :] * matrices_in[1, 1, :])
        
        return matrices_out
        
    def resample_systematic(self, weights: np.ndarray, num_particles: int) -> np.ndarray:
        """
        Systematic resampling of particles.
        Port of resampleSystematic.m
        
        Args:
            weights: Normalized particle weights (num_particles,)
            num_particles: Number of particles to resample
            
        Returns:
            indexes: Resampled particle indices (num_particles,)
        """
        indexes = np.zeros(num_particles, dtype=int)
        cum_weights = np.cumsum(weights)
        
        # Create systematic grid
        u = np.random.rand() / num_particles
        grid = np.linspace(u, 1 - 1/num_particles + u, num_particles)
        
        i = 0
        j = 0
        while i < num_particles:
            if grid[i] < cum_weights[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
                
        return indexes
        
    def data_association_bp(self, input_da: np.ndarray) -> np.ndarray:
        """
        Belief propagation for data association.
        Port of dataAssociationBP.m
        
        Args:
            input_da: Input association probabilities (2, num_targets)
            
        Returns:
            output_da: Output association probabilities (num_targets,)
        """
        # Compute association ratios
        ratios = input_da[1, :] / input_da[0, :]
        sum_ratios = 1 + np.sum(ratios)
        
        output_da = 1.0 / (sum_ratios - ratios)
        
        # Handle NaN cases with hard assignment
        if np.any(np.isnan(output_da)):
            output_da = np.zeros_like(output_da)
            max_idx = np.argmax(ratios)
            output_da[max_idx] = 1.0
            
        return output_da
        
    def get_weights_unknown(self, log_weights: np.ndarray, 
                           old_existence: float, skip_index: int) -> Tuple[np.ndarray, float]:
        """
        Compute weights and updated existence probability.
        Port of getWeightsUnknown.m
        
        Args:
            log_weights: Log weights matrix (num_particles, num_measurements)
            old_existence: Previous existence probability
            skip_index: Index to skip (0-based, -1 for no skip)
            
        Returns:
            weights: Normalized weights (num_particles,)
            updated_existence: Updated existence probability
        """
        if skip_index >= 0:
            log_weights[:, skip_index] = 0
            
        log_weights_sum = np.sum(log_weights, axis=1)
        
        alive_update = np.mean(np.exp(log_weights_sum))
        
        if np.isinf(alive_update):
            updated_existence = 1.0
        else:
            alive = old_existence * alive_update
            dead = 1 - old_existence
            updated_existence = alive / (dead + alive)
            
        # Normalize weights
        weights = np.exp(log_weights_sum - np.max(log_weights_sum))
        weights = weights / np.sum(weights)
        
        return weights, updated_existence
        
    def update_particles(self, old_particles_kinematic: np.ndarray,
                        old_particles_extent: np.ndarray, 
                        old_existence: float,
                        log_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Update particle set using importance sampling.
        Port of updateParticles.m
        
        Args:
            old_particles_kinematic: Old kinematic particles (4, num_particles)
            old_particles_extent: Old extent particles (2, 2, num_particles)
            old_existence: Old existence probability
            log_weights: Log weights matrix (num_particles, num_measurements)
            
        Returns:
            updated_particles_kinematic: Updated kinematic particles
            updated_particles_extent: Updated extent particles  
            updated_existence: Updated existence probability
        """
        num_particles = self.parameters['numParticles']
        regularization_deviation = self.parameters['regularizationDeviation']
        
        log_weights_sum = np.sum(log_weights, axis=1)
        
        alive_update = np.mean(np.exp(log_weights_sum))
        
        if np.isinf(alive_update):
            updated_existence = 1.0
        else:
            alive = old_existence * alive_update
            dead = 1 - old_existence
            updated_existence = alive / (dead + alive)
            
        if updated_existence != 0:
            # Normalize weights and resample
            log_weights_norm = log_weights_sum - np.max(log_weights_sum)
            weights = np.exp(log_weights_norm)
            weights_normalized = weights / np.sum(weights)
            
            indexes = self.resample_systematic(weights_normalized, num_particles)
            
            updated_particles_kinematic = old_particles_kinematic[:, indexes]
            updated_particles_extent = old_particles_extent[:, :, indexes]
            
            # Add regularization noise
            updated_particles_kinematic[:2, :] += (regularization_deviation * 
                                                  np.random.randn(2, num_particles))
        else:
            updated_particles_kinematic = np.full_like(old_particles_kinematic, np.nan)
            updated_particles_extent = np.full_like(old_particles_extent, np.nan)
            
        return updated_particles_kinematic, updated_particles_extent, updated_existence
        
    def iwishrnd_fast_vector(self, scale_matrix: np.ndarray, 
                            degrees_freedom: float, 
                            num_samples: int) -> np.ndarray:
        """
        Fast inverse Wishart sampling for multiple samples.
        
        Args:
            scale_matrix: Scale matrix (2, 2)
            degrees_freedom: Degrees of freedom
            num_samples: Number of samples
            
        Returns:
            samples: Inverse Wishart samples (2, 2, num_samples)
        """
        samples = np.zeros((2, 2, num_samples))
        
        for i in range(num_samples):
            # Sample from Wishart distribution
            w_sample = wishart.rvs(df=degrees_freedom, scale=scale_matrix)
            # Inverse for inverse Wishart
            samples[:, :, i] = np.linalg.inv(w_sample)
            
        return samples
        
    def perform_prediction(self, current_particles_kinematic: np.ndarray,
                          current_existences: np.ndarray,
                          current_particles_extent: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform prediction step for all targets.
        Port of performPrediction.m
        
        Args:
            current_particles_kinematic: Current kinematic particles (4, num_particles, num_targets)
            current_existences: Current existence probabilities (num_targets,)
            current_particles_extent: Current extent particles (2, 2, num_particles, num_targets)
            
        Returns:
            predicted_particles_kinematic: Predicted kinematic particles
            predicted_existences: Predicted existence probabilities
            predicted_particles_extent: Predicted extent particles
        """
        scan_time = self.parameters['scanTime']
        acceleration_deviation = self.parameters['accelerationDeviation']
        survival_probability = self.parameters['survivalProbability']
        degree_freedom_prediction = self.parameters['degreeFreedomPrediction']
        
        # Get transition matrices
        A = np.eye(4)
        A[0, 2] = scan_time
        A[1, 3] = scan_time
        
        W = np.zeros((4, 2))
        W[0, 0] = 0.5 * scan_time**2
        W[1, 1] = 0.5 * scan_time**2
        W[2, 0] = scan_time
        W[3, 1] = scan_time
        
        num_targets = current_particles_kinematic.shape[2]
        num_particles = current_particles_kinematic.shape[1]
        
        # Predict kinematic states
        predicted_particles_kinematic = np.zeros_like(current_particles_kinematic)
        for target in range(num_targets):
            predicted_particles_kinematic[:, :, target] = (
                A @ current_particles_kinematic[:, :, target] +
                W @ (acceleration_deviation * np.random.randn(2, num_particles))
            )
            
        # Predict existence probabilities
        predicted_existences = survival_probability * current_existences
        
        # Predict extent particles
        predicted_particles_extent = np.zeros_like(current_particles_extent)
        for target in range(num_targets):
            # Scale by degree of freedom for prediction
            scaled_extents = current_particles_extent[:, :, :, target] / degree_freedom_prediction
            predicted_particles_extent[:, :, :, target] = self.iwishrnd_fast_vector(
                scaled_extents[:, :, 0], degree_freedom_prediction, num_particles
            )
            
        return predicted_particles_kinematic, predicted_existences, predicted_particles_extent
        
    def track_formation(self, estimates: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Form tracks from frame-by-frame estimates.
        Port of trackFormation.m
        
        Args:
            estimates: List of estimates for each time step
            
        Returns:
            tracks: Formed tracks (4, num_steps, num_tracks)
            extents: Track extents (2, 2, num_steps, num_tracks)
        """
        num_steps = len(estimates)
        minimum_track_length = self.parameters['minimumTrackLength']
        
        # Collect all labels
        all_labels = []
        for step_estimates in estimates:
            if step_estimates:
                for est in step_estimates:
                    all_labels.append(tuple(est['label']))
                    
        unique_labels = list(set(all_labels))
        num_tracks = len(unique_labels)
        
        # Initialize track arrays
        tracks = np.full((4, num_steps, num_tracks), np.nan)
        extents = np.full((2, 2, num_steps, num_tracks), np.nan)
        
        # Fill tracks
        for step, step_estimates in enumerate(estimates):
            if step_estimates:
                for est in step_estimates:
                    label_tuple = tuple(est['label'])
                    track_idx = unique_labels.index(label_tuple)
                    tracks[:, step, track_idx] = est['state']
                    extents[:, :, step, track_idx] = est['extent']
        
        # Remove short tracks
        track_lengths = np.sum(~np.isnan(tracks[0, :, :]), axis=0)
        valid_tracks = track_lengths >= minimum_track_length
        
        tracks = tracks[:, :, valid_tracks]
        extents = extents[:, :, :, valid_tracks]
        
        return tracks, extents
        
    def eot_elliptical_shape(self, measurements_cell: List[np.ndarray]) -> List[List[Dict]]:
        """
        Main Extended Object Tracking algorithm using belief propagation.
        Port of eotEllipticalShape.m
        
        Args:
            measurements_cell: List of measurement arrays for each time step
            
        Returns:
            estimates: List of estimates for each time step
        """
        # Extract parameters
        num_particles = self.parameters['numParticles']
        mean_clutter = self.parameters['meanClutter']
        mean_measurements = self.parameters['meanMeasurements']
        scan_time = self.parameters['scanTime']
        detection_threshold = self.parameters['detectionThreshold']
        threshold_pruning = self.parameters['thresholdPruning']
        num_outer_iterations = self.parameters['numOuterIterations']
        mean_births = self.parameters['meanBirths']
        prior_velocity_covariance = self.parameters['priorVelocityCovariance']
        surveillance_region = self.parameters['surveillanceRegion']
        measurement_variance = self.parameters['measurementVariance']
        prior_extent1 = self.parameters['priorExtent1']
        prior_extent2 = self.parameters['priorExtent2']
        
        # surveillance_region format: [[x_min, y_min], [x_max, y_max]]
        area_size = ((surveillance_region[1, 0] - surveillance_region[0, 0]) * 
                    (surveillance_region[1, 1] - surveillance_region[0, 1]))
        measurements_covariance = measurement_variance * np.eye(2)
        
        # Ensure area_size is positive to avoid division by zero
        if area_size <= 0:
            raise ValueError(f"Invalid surveillance region area: {area_size}, region: {surveillance_region}")
        
        mean_extent_prior = (prior_extent1 / (prior_extent2 - 3))
        mean_extent_prior_squared = mean_extent_prior @ mean_extent_prior.T
        total_covariance = mean_extent_prior_squared + measurements_covariance
        
        num_steps = len(measurements_cell)
        constant_factor = area_size * (mean_measurements / mean_clutter)
        uniform_weight = np.log(1.0 / area_size)
        
        estimates = []
        current_labels = np.zeros((2, 0), dtype=int)
        current_particles_kinematic = np.zeros((4, num_particles, 0))
        current_existences = np.zeros(0)
        current_particles_extent = np.zeros((2, 2, num_particles, 0))
        
        for step in range(num_steps):
            print(f"Processing step {step + 1}")
            
            measurements = measurements_cell[step]
            num_measurements = measurements.shape[1]
            
            # Prediction step
            if current_particles_kinematic.shape[2] > 0:
                (current_particles_kinematic, current_existences, 
                 current_particles_extent) = self.perform_prediction(
                    current_particles_kinematic, current_existences, current_particles_extent
                )
                
                # Update existence probabilities
                current_alive = current_existences * np.exp(-mean_measurements)
                current_dead = 1 - current_existences
                current_existences = current_alive / (current_dead + current_alive)
                
            num_targets = current_particles_kinematic.shape[2]
            num_legacy = num_targets
            
            # Get promising new targets (simplified version)
            new_indexes = np.arange(min(3, num_measurements))  # Simplified: take first 3 measurements
            num_new = len(new_indexes)
            
            if num_new > 0:
                new_labels = np.array([[step + 1] * num_new, new_indexes])
                current_labels = np.concatenate([current_labels, new_labels], axis=1)
                
                # Initialize new targets
                new_existences = np.full(num_new, 
                    mean_births * np.exp(-mean_measurements) / 
                    (mean_births * np.exp(-mean_measurements) + 1)
                )
                
                new_particles_kinematic = np.zeros((4, num_particles, num_new))
                new_particles_extent = np.zeros((2, 2, num_particles, num_new))
                
                for target in range(num_new):
                    proposal_mean = measurements[:, new_indexes[target]]
                    proposal_covariance = 2 * total_covariance
                    
                    # Sample positions
                    new_particles_kinematic[:2, :, target] = (
                        proposal_mean[:, np.newaxis] + 
                        np.random.multivariate_normal(
                            np.zeros(2), proposal_covariance, num_particles
                        ).T
                    )
                    
                    # Sample extents
                    new_particles_extent[:, :, :, target] = self.iwishrnd_fast_vector(
                        prior_extent1, prior_extent2, num_particles
                    )
                
                # Concatenate new targets
                current_existences = np.concatenate([current_existences, new_existences])
                current_particles_kinematic = np.concatenate([
                    current_particles_kinematic, new_particles_kinematic
                ], axis=2)
                current_particles_extent = np.concatenate([
                    current_particles_extent, new_particles_extent
                ], axis=3)
            
            # Simplified belief propagation (reduced complexity)
            num_targets = current_particles_kinematic.shape[2]
            
            # Update all targets
            for target in range(num_targets):
                if num_measurements > 0:
                    log_weights = np.zeros((num_particles, num_measurements))
                    for measurement in range(num_measurements):
                        extent_cov = (self.get_square2_fast(current_particles_extent[:, :, :, target]) + 
                                    measurements_covariance[:, :, np.newaxis])
                        log_weights[:, measurement] = self.get_log_weights_fast(
                            measurements[:, measurement],
                            current_particles_kinematic[:2, :, target],
                            extent_cov
                        )
                    
                    # Update particles
                    (current_particles_kinematic[:, :, target], 
                     current_particles_extent[:, :, :, target],
                     current_existences[target]) = self.update_particles(
                        current_particles_kinematic[:, :, target],
                        current_particles_extent[:, :, :, target],
                        current_existences[target],
                        log_weights
                    )
            
            # Pruning
            valid_targets = current_existences >= threshold_pruning
            current_particles_kinematic = current_particles_kinematic[:, :, valid_targets]
            current_particles_extent = current_particles_extent[:, :, :, valid_targets]
            current_labels = current_labels[:, valid_targets]
            current_existences = current_existences[valid_targets]
            
            # Estimation
            step_estimates = []
            num_targets = current_particles_kinematic.shape[2]
            
            for target in range(num_targets):
                if current_existences[target] > detection_threshold:
                    estimate = {
                        'state': np.mean(current_particles_kinematic[:, :, target], axis=1),
                        'extent': np.mean(current_particles_extent[:, :, :, target], axis=2),
                        'label': current_labels[:, target]
                    }
                    step_estimates.append(estimate)
                    
            estimates.append(step_estimates)
            
        return estimates
        
    def run_filter(self, measurements: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the complete Extended Object Filter pipeline.
        
        Args:
            measurements: List of measurement arrays for each time step
            
        Returns:
            estimated_tracks: Estimated target tracks
            estimated_extents: Estimated target extents
        """
        # Run main filtering algorithm
        estimates = self.eot_elliptical_shape(measurements)
        
        # Form tracks from estimates
        estimated_tracks, estimated_extents = self.track_formation(estimates)
        
        return estimated_tracks, estimated_extents