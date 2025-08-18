"""
Extended Object Tracking Filter Implementation

This module contains the EO_Filter class that implements the main Extended Object 
Tracking algorithm using belief propagation and particle filtering.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from scipy.stats import multivariate_normal, poisson, wishart, invwishart
from scipy.linalg import sqrtm
from dataclasses import dataclass
from GenData import EOTParameters


@dataclass
class TrackEstimate:
    """Structure to hold track estimates for a single time step."""
    state: np.ndarray      # Kinematic states (4 x num_detections)
    extent: np.ndarray     # Extent matrices (2 x 2 x num_detections)  
    label: np.ndarray      # Track labels (2 x num_detections)


class EO_Filter:
    """
    Extended Object Filter class implementing the main EOT tracking algorithm.
    
    This class handles:
    - Belief propagation for data association
    - Particle filtering for state estimation
    - Track management (birth/death detection)
    - Extent estimation using inverse Wishart models
    """
    
    def __init__(self, parameters: EOTParameters):
        """
        Initialize the EO Filter with tracking parameters.
        
        Args:
            parameters: EOT parameters containing algorithm settings
        """
        self.parameters = parameters
        
        # Precompute derived parameters
        self.area_size = ((parameters.surveillance_region[0, 1] - parameters.surveillance_region[0, 0]) * 
                         (parameters.surveillance_region[1, 1] - parameters.surveillance_region[1, 0]))
        self.measurements_covariance = parameters.measurement_variance * np.eye(2)
        
        mean_extent_prior = np.trace(parameters.prior_extent_1 / (parameters.prior_extent_2 - 3))
        self.total_covariance = mean_extent_prior**2 + self.measurements_covariance
        
        self.constant_factor = self.area_size * (parameters.mean_measurements / parameters.mean_clutter)
        self.uniform_weight = np.log(1.0 / self.area_size)

    def track(self, measurements_cell: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main tracking function that processes a sequence of measurements.
        
        Args:
            measurements_cell: List of measurement arrays, one per time step
            
        Returns:
            estimated_tracks: Estimated kinematic trajectories (4 x num_steps x num_tracks)
            estimated_extents: Estimated extent trajectories (2 x 2 x num_steps x num_tracks)
        """
        num_steps = len(measurements_cell)
        
        # Initialize tracking state
        estimates = [None] * num_steps
        current_labels = np.empty((2, 0))  # Track labels [birth_time; measurement_index]
        current_particles_kinematic = np.empty((4, self.parameters.num_particles, 0))  # Kinematic particles
        current_existences = np.empty(0)  # Existence probabilities
        current_particles_extent = np.empty((2, 2, self.parameters.num_particles, 0))  # Extent particles
        
        # Process each time step
        for step in range(num_steps):
            print(f"Processing time step {step + 1}/{num_steps}")
            
            # Load measurements for current time step
            measurements = measurements_cell[step]
            num_measurements = measurements.shape[1] if measurements.size > 0 else 0
            
            # Prediction step
            (current_particles_kinematic, current_existences, 
             current_particles_extent) = self._perform_prediction(
                current_particles_kinematic, current_existences,
                current_particles_extent
            )
            
            # Update existence probabilities
            current_alive = current_existences * np.exp(-self.parameters.mean_measurements)
            current_dead = 1 - current_existences
            current_existences = current_alive / (current_dead + current_alive)
            
            num_legacy = current_particles_kinematic.shape[2]
            
            # Birth detection
            if num_measurements > 0:
                new_indexes, measurements = self._get_promising_new_targets(
                    current_particles_kinematic, current_particles_extent,
                    current_existences, measurements
                )
            else:
                new_indexes = np.array([])
                
            num_new = len(new_indexes)
            
            # Add labels for new potential targets
            if num_new > 0:
                new_labels = np.array([
                    [step + 1] * num_new,  # Birth time (1-indexed)
                    new_indexes            # Measurement index  
                ])
                current_labels = np.concatenate([current_labels, new_labels], axis=1)
            
            # Initialize new targets
            (current_particles_kinematic, current_particles_extent, current_existences,
             weights_extrinsic, weights_extrinsic_new, likelihood_1, likelihood_new_1, new_weights) = self._initialize_new_targets(
                current_particles_kinematic, current_particles_extent, current_existences,
                new_indexes, measurements, num_legacy, num_new
            )
            
            current_existences_extrinsic = np.tile(current_existences, (num_measurements, 1)).T
            
            # Belief propagation iterations
            for outer in range(self.parameters.num_outer_iterations):
                output_da, target_indexes = self._belief_propagation_iteration(
                    measurements, num_measurements, num_legacy, num_new, new_indexes,
                    current_particles_kinematic, current_particles_extent, 
                    current_existences_extrinsic, weights_extrinsic, weights_extrinsic_new,
                    likelihood_1, likelihood_new_1, new_weights, outer
                )
                
                # Update particles and existence probabilities
                (current_particles_kinematic, current_particles_extent, 
                 current_existences, weights_extrinsic, weights_extrinsic_new) = self._update_step(
                    current_particles_kinematic, current_particles_extent, current_existences,
                    output_da, target_indexes, measurements, num_measurements, num_legacy, num_new,
                    new_indexes, likelihood_1, likelihood_new_1, new_weights, outer, weights_extrinsic, weights_extrinsic_new
                )
                
                current_existences_extrinsic = np.tile(current_existences, (num_measurements, 1)).T
            
            # Pruning
            (current_particles_kinematic, current_particles_extent, 
             current_labels, current_existences) = self._pruning_step(
                current_particles_kinematic, current_particles_extent,
                current_labels, current_existences
            )
            
            # Estimation
            estimates[step] = self._extract_estimates(
                current_particles_kinematic, current_particles_extent,
                current_labels, current_existences
            )
        
        # Track formation
        estimated_tracks, estimated_extents = self._track_formation(estimates)
        
        return estimated_tracks, estimated_extents

    def _perform_prediction(self, old_particles: np.ndarray, old_existences: np.ndarray,
                           old_extents: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform prediction step for kinematic states and extent matrices."""
        if old_particles.shape[2] == 0:
            return old_particles, old_existences, old_extents
        
        num_particles = old_particles.shape[1]
        num_targets = old_particles.shape[2]
        
        # Get motion model matrices
        A, W = self._get_transition_matrices()
        
        # Initialize output arrays
        new_particles = old_particles.copy()
        new_existences = old_existences.copy()
        new_extents = old_extents.copy()
        
        # Predict extent matrices using Wishart evolution
        for target in range(num_targets):
            scale_matrices = old_extents[:, :, :, target] / self.parameters.degree_freedom_prediction
            new_extents[:, :, :, target] = self._wishart_fast_vector(
                scale_matrices, self.parameters.degree_freedom_prediction, num_particles
            )
        
        # Predict kinematic states and existences
        for target in range(num_targets):
            # Kinematic prediction with process noise
            process_noise = W @ (np.sqrt(self.parameters.acceleration_deviation**2) * 
                               np.random.randn(2, num_particles))
            new_particles[:, :, target] = A @ old_particles[:, :, target] + process_noise
            
            # Existence prediction
            new_existences[target] = self.parameters.survival_probability * old_existences[target]
        
        return new_particles, new_existences, new_extents

    def _get_transition_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate state transition matrices for motion model."""
        scan_time = self.parameters.scan_time
        
        # State transition matrix for [x, y, vx, vy]
        A = np.eye(4)
        A[0, 2] = scan_time  # x = x + vx * dt
        A[1, 3] = scan_time  # y = y + vy * dt
        
        # Process noise input matrix
        W = np.zeros((4, 2))
        W[0, 0] = 0.5 * scan_time**2
        W[1, 1] = 0.5 * scan_time**2  
        W[2, 0] = scan_time
        W[3, 1] = scan_time
        
        return A, W

    def _get_promising_new_targets(self, current_particles_kinematic: np.ndarray, 
                                  current_particles_extent: np.ndarray,
                                  current_existences: np.ndarray,
                                  measurements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Identify measurements likely to originate from new targets."""
        if measurements.size == 0:
            return np.array([]), measurements
        
        num_measurements = measurements.shape[1]
        
        if current_particles_kinematic.shape[2] == 0:
            return np.arange(num_measurements), measurements
        
        # Compute distance from each measurement to existing target particles
        min_distances = np.full(num_measurements, np.inf)
        
        for measurement_idx in range(num_measurements):
            measurement = measurements[:, measurement_idx]
            
            for target in range(current_particles_kinematic.shape[2]):
                if current_existences[target] > 0.1:
                    target_positions = current_particles_kinematic[:2, :, target]
                    distances = np.linalg.norm(
                        target_positions - measurement.reshape(-1, 1), axis=0
                    )
                    min_distances[measurement_idx] = min(
                        min_distances[measurement_idx], np.min(distances)
                    )
        
        # Measurements far from existing targets are potential new births
        birth_threshold = 20.0
        promising_indexes = np.where(min_distances > birth_threshold)[0]
        
        # Limit number of new births per time step
        max_new_births = min(5, len(promising_indexes))
        if len(promising_indexes) > max_new_births:
            promising_indexes = promising_indexes[:max_new_births]
        
        return promising_indexes, measurements

    def _initialize_new_targets(self, current_particles_kinematic, current_particles_extent, 
                               current_existences, new_indexes, measurements, num_legacy, num_new):
        """Initialize particles for new potential targets."""
        # Initialize belief propagation variables
        new_existences = np.full(num_new, self.parameters.mean_births * np.exp(-self.parameters.mean_measurements) / 
                                (self.parameters.mean_births * np.exp(-self.parameters.mean_measurements) + 1))
        new_particles_kinematic = np.zeros((4, self.parameters.num_particles, num_new))
        new_particles_extent = np.zeros((2, 2, self.parameters.num_particles, num_new))
        new_weights = np.zeros((self.parameters.num_particles, num_new))
        
        # Initialize each new target
        for target in range(num_new):
            if new_indexes[target] < measurements.shape[1]:
                # Proposal distribution centered at measurement location
                proposal_mean = measurements[:, int(new_indexes[target])]
                proposal_covariance = 2 * self.total_covariance
                
                # Sample position particles from proposal
                pos_samples = (proposal_mean.reshape(-1, 1) + 
                              sqrtm(proposal_covariance) @ np.random.randn(2, self.parameters.num_particles))
                new_particles_kinematic[:2, :, target] = pos_samples
                
                # Compute importance weights
                for p in range(self.parameters.num_particles):
                    weight = (self.uniform_weight - 
                             multivariate_normal.logpdf(
                                 new_particles_kinematic[:2, p, target],
                                 proposal_mean, proposal_covariance))
                    new_weights[p, target] = weight
                
                # Sample extent particles from prior
                new_particles_extent[:, :, :, target] = self._iwishart_fast_vector(
                    self.parameters.prior_extent_1, self.parameters.prior_extent_2, self.parameters.num_particles
                )
        
        # Combine legacy and new targets
        current_existences = np.concatenate([current_existences, new_existences])
        current_particles_kinematic = np.concatenate([
            current_particles_kinematic, new_particles_kinematic], axis=2)
        current_particles_extent = np.concatenate([
            current_particles_extent, new_particles_extent], axis=3)
        
        # Initialize message passing arrays
        num_measurements = measurements.shape[1] if measurements.size > 0 else 0
        weights_extrinsic = np.full((self.parameters.num_particles, num_measurements, num_legacy), np.nan)
        weights_extrinsic_new = np.full((self.parameters.num_particles, num_measurements, num_new), np.nan)
        
        likelihood_1 = np.zeros((self.parameters.num_particles, num_measurements, num_legacy + num_new))
        likelihood_new_1 = np.full((self.parameters.num_particles, num_measurements, num_new), np.nan)
        
        return (current_particles_kinematic, current_particles_extent, current_existences,
                weights_extrinsic, weights_extrinsic_new, likelihood_1, likelihood_new_1, new_weights)

    def _belief_propagation_iteration(self, measurements, num_measurements, num_legacy, num_new, new_indexes,
                                     current_particles_kinematic, current_particles_extent, 
                                     current_existences_extrinsic, weights_extrinsic, weights_extrinsic_new,
                                     likelihood_1, likelihood_new_1, new_weights, outer):
        """Perform one iteration of belief propagation for data association."""
        output_da = [None] * num_measurements
        target_indexes = [None] * num_measurements
        
        # Process measurements in reverse order
        for measurement in range(num_measurements - 1, -1, -1):
            # Initialize data association input
            input_da = np.ones((2, num_legacy + num_new))
            
            # Process legacy targets
            for target in range(num_legacy):
                if outer == 0:
                    # First iteration: compute likelihoods
                    extent_matrices = self._get_square_2_fast(current_particles_extent[:, :, :, target])
                    meas_cov_tiled = np.tile(self.measurements_covariance[:, :, np.newaxis], 
                                           (1, 1, self.parameters.num_particles))
                    total_cov = extent_matrices + meas_cov_tiled
                    
                    likelihood_1[:, measurement, target] = (
                        self.constant_factor * np.exp(self._get_log_weights_fast(
                            measurements[:, measurement],
                            current_particles_kinematic[:2, :, target],
                            total_cov
                        ))
                    )
                    input_da[1, target] = (current_existences_extrinsic[target, measurement] * 
                                          np.mean(likelihood_1[:, measurement, target]))
                else:
                    # Use extrinsic information from previous iteration
                    input_da[1, target] = (current_existences_extrinsic[target, measurement] *
                                          np.dot(weights_extrinsic[:, measurement, target],
                                                likelihood_1[:, measurement, target]))
                
                input_da[0, target] = 1
            
            # Process new targets
            target_index = num_legacy
            target_indexes_current = []
            
            for target_meas in range(num_measurements - 1, measurement - 1, -1):
                if target_meas in new_indexes:
                    target_idx_in_new = np.where(new_indexes == target_meas)[0][0]
                    target_indexes_current.append(target_meas)
                    
                    if outer == 0:
                        # First iteration
                        weights = np.exp(new_weights[:, target_idx_in_new])
                        weights = weights / np.sum(weights)
                        
                        extent_matrices = self._get_square_2_fast(current_particles_extent[:, :, :, target_index])
                        meas_cov_tiled = np.tile(self.measurements_covariance[:, :, np.newaxis], 
                                               (1, 1, self.parameters.num_particles))
                        total_cov = extent_matrices + meas_cov_tiled
                        
                        likelihood_new_1[:, measurement, target_idx_in_new] = (
                            self.constant_factor * np.exp(self._get_log_weights_fast(
                                measurements[:, measurement],
                                current_particles_kinematic[:2, :, target_index],
                                total_cov
                            ))
                        )
                        input_da[1, target_index] = (
                            current_existences_extrinsic[target_index, measurement] *
                            np.dot(weights, likelihood_new_1[:, measurement, target_idx_in_new])
                        )
                    else:
                        # Use extrinsic information
                        input_da[1, target_index] = (
                            current_existences_extrinsic[target_index, measurement] *
                            np.dot(weights_extrinsic_new[:, measurement, target_idx_in_new],
                                  likelihood_new_1[:, measurement, target_idx_in_new])
                        )
                    
                    input_da[0, target_index] = 1
                    
                    # Special case: measurement can be unassociated to its own new target
                    if target_meas == measurement:
                        input_da[0, target_index] = 1 - current_existences_extrinsic[target_index, measurement]
                    
                    target_index += 1
            
            target_indexes[measurement] = target_indexes_current
            output_da[measurement] = self._data_association_bp(input_da[:, :target_index])
        
        return output_da, target_indexes

    def _update_step(self, current_particles_kinematic, current_particles_extent, current_existences,
                    output_da, target_indexes, measurements, num_measurements, num_legacy, num_new,
                    new_indexes, likelihood_1, likelihood_new_1, new_weights, outer, weights_extrinsic, weights_extrinsic_new):
        """Update particles and existence probabilities based on measurement evidence."""
        # Update legacy targets
        for target in range(num_legacy):
            weights = np.zeros((self.parameters.num_particles, num_measurements))
            
            for measurement in range(num_measurements):
                if output_da[measurement] is not None and target < output_da[measurement].shape[0]:
                    current_weights = (1 + 
                                     likelihood_1[:, measurement, target] * 
                                     output_da[measurement][target])
                    # Add small epsilon to prevent log(0)
                    current_weights = np.maximum(current_weights, 1e-300)
                    weights[:, measurement] = np.log(current_weights)
            
            # Compute extrinsic information or final belief
            if outer != self.parameters.num_outer_iterations - 1:
                # Not last iteration: compute extrinsic information
                for measurement in range(num_measurements):
                    (weights_extrinsic[:, measurement, target],
                     _) = self._get_weights_unknown(
                        weights, current_existences[target], measurement
                    )
            else:
                # Last iteration: update particles
                (current_particles_kinematic[:, :, target],
                 current_particles_extent[:, :, :, target],
                 current_existences[target]) = self._update_particles(
                    current_particles_kinematic[:, :, target],
                    current_particles_extent[:, :, :, target],
                    current_existences[target], weights
                )
        
        # Update new targets
        target_index = num_legacy
        for target_meas in range(num_measurements - 1, -1, -1):
            if target_meas in new_indexes:
                target_idx_in_new = np.where(new_indexes == target_meas)[0][0]
                
                weights = np.zeros((self.parameters.num_particles, num_measurements + 1))
                weights[:, num_measurements] = new_weights[:, target_idx_in_new]
                
                for measurement in range(target_meas + 1):
                    if (output_da[measurement] is not None and 
                        target_meas in target_indexes[measurement]):
                        
                        target_pos_in_da = (len([t for t in target_indexes[measurement] 
                                               if t == target_meas]) - 1)
                        if target_pos_in_da < output_da[measurement].shape[0]:
                            output_tmp_da = output_da[measurement][num_legacy + target_pos_in_da]
                            
                            if not np.isinf(output_tmp_da):
                                current_weights = (likelihood_new_1[:, measurement, target_idx_in_new] * 
                                                 output_tmp_da)
                            else:
                                current_weights = likelihood_new_1[:, measurement, target_idx_in_new]
                            
                            if measurement != target_meas:
                                current_weights = current_weights + 1
                            
                            # Add small epsilon to prevent log(0)
                            current_weights = np.maximum(current_weights, 1e-300)
                            weights[:, measurement] = np.log(current_weights)
                
                # Compute extrinsic information or final belief
                if outer != self.parameters.num_outer_iterations - 1:
                    # Not last iteration
                    for measurement in range(target_meas + 1):
                        (weights_extrinsic_new[:, measurement, target_idx_in_new],
                         _) = self._get_weights_unknown(
                            weights, current_existences[target_index], measurement
                        )
                else:
                    # Last iteration: update particles
                    (current_particles_kinematic[:2, :, target_index],
                     current_particles_extent[:, :, :, target_index],
                     current_existences[target_index]) = self._update_particles(
                        current_particles_kinematic[:2, :, target_index],
                        current_particles_extent[:, :, :, target_index],
                        current_existences[target_index], weights
                    )
                    
                    # Sample velocity for new targets
                    current_particles_kinematic[2:4, :, target_index] = (
                        np.random.multivariate_normal([0, 0], self.parameters.prior_velocity_covariance, 
                                                    self.parameters.num_particles).T
                    )
                
                target_index += 1
        
        return (current_particles_kinematic, current_particles_extent, current_existences,
                weights_extrinsic, weights_extrinsic_new)

    def _pruning_step(self, current_particles_kinematic, current_particles_extent,
                     current_labels, current_existences):
        """Remove low-probability tracks."""
        is_redundant = current_existences < self.parameters.threshold_pruning
        
        # Keep only non-redundant tracks
        current_particles_kinematic = current_particles_kinematic[:, :, ~is_redundant]
        current_particles_extent = current_particles_extent[:, :, :, ~is_redundant]
        current_labels = current_labels[:, ~is_redundant]
        current_existences = current_existences[~is_redundant]
        
        return current_particles_kinematic, current_particles_extent, current_labels, current_existences

    def _extract_estimates(self, current_particles_kinematic, current_particles_extent,
                          current_labels, current_existences):
        """Extract state estimates for detected targets."""
        num_targets = current_particles_kinematic.shape[2]
        detected_targets = 0
        current_estimate = TrackEstimate(
            state=np.empty((4, 0)),
            extent=np.empty((2, 2, 0)), 
            label=np.empty((2, 0))
        )
        
        for target in range(num_targets):
            if current_existences[target] > self.parameters.detection_threshold:
                # Compute state estimate as particle mean
                state_est = np.mean(current_particles_kinematic[:, :, target], axis=1)
                extent_est = np.mean(current_particles_extent[:, :, :, target], axis=2)
                label_est = current_labels[:, target]
                
                # Add to estimates
                current_estimate.state = np.column_stack([current_estimate.state, state_est])
                current_estimate.extent = np.concatenate([current_estimate.extent, 
                                                        extent_est[:, :, np.newaxis]], axis=2)
                current_estimate.label = np.column_stack([current_estimate.label, label_est])
                detected_targets += 1
        
        return current_estimate if detected_targets > 0 else None

    def _track_formation(self, estimates: List[TrackEstimate]) -> Tuple[np.ndarray, np.ndarray]:
        """Form continuous tracks from frame-by-frame estimates."""
        num_steps = len(estimates)
        
        # Collect all unique labels
        all_labels = []
        for step in range(num_steps):
            if estimates[step] is not None:
                current_labels = estimates[step].label
                if current_labels.size > 0:
                    for i in range(current_labels.shape[1]):
                        label = tuple(current_labels[:, i])
                        if label not in all_labels:
                            all_labels.append(label)
        
        if len(all_labels) == 0:
            return np.empty((4, num_steps, 0)), np.empty((2, 2, num_steps, 0))
        
        labels_array = np.array(all_labels).T
        num_tracks = labels_array.shape[1]
        
        # Initialize track arrays
        tracks = np.full((4, num_steps, num_tracks), np.nan)
        extents = np.full((2, 2, num_steps, num_tracks), np.nan)
        
        # Fill tracks with estimates
        for step in range(num_steps):
            if estimates[step] is not None:
                current_labels = estimates[step].label
                current_states = estimates[step].state
                current_extents = estimates[step].extent
                
                num_estimates = current_labels.shape[1]
                for estimate in range(num_estimates):
                    estimate_label = current_labels[:, estimate]
                    
                    # Find matching track
                    for track in range(num_tracks):
                        if np.array_equal(estimate_label, labels_array[:, track]):
                            tracks[:, step, track] = current_states[:, estimate]
                            extents[:, :, step, track] = current_extents[:, :, estimate]
                            break
        
        # Remove tracks that are too short
        valid_tracks = []
        for track in range(num_tracks):
            track_length = np.sum(~np.isnan(tracks[0, :, track]))
            if track_length >= self.parameters.minimum_track_length:
                valid_tracks.append(track)
        
        if len(valid_tracks) > 0:
            tracks = tracks[:, :, valid_tracks]
            extents = extents[:, :, :, valid_tracks]
        else:
            tracks = np.empty((4, num_steps, 0))
            extents = np.empty((2, 2, num_steps, 0))
        
        return tracks, extents

    # Helper functions
    def _data_association_bp(self, input_da: np.ndarray) -> np.ndarray:
        """Belief propagation for data association."""
        if input_da.shape[1] == 0:
            return np.array([])
        
        input_ratios = input_da[1, :] / input_da[0, :]
        sum_input = 1 + np.sum(input_ratios)
        output_da = 1.0 / (sum_input - input_ratios)
        
        # Handle NaN cases with hard association
        if np.any(np.isnan(output_da)):
            output_da = np.zeros_like(output_da)
            max_idx = np.argmax(input_ratios)
            output_da[max_idx] = 1.0
        
        return output_da

    def _get_log_weights_fast(self, measurement: np.ndarray, particles_kinematic: np.ndarray,
                             particles_extent: np.ndarray) -> np.ndarray:
        """Fast computation of log-likelihood weights."""
        # Compute determinants efficiently
        det_11 = particles_extent[0, 0, :]
        det_22 = particles_extent[1, 1, :]  
        det_12 = particles_extent[0, 1, :]
        determinants = det_11 * det_22 - det_12**2
        
        # Normalization factors
        log_factors = np.log(1.0 / (2 * np.pi * np.sqrt(determinants)))
        
        # Innovation vectors
        innovations = measurement.reshape(-1, 1) - particles_kinematic[:2, :]
        
        # Efficient computation of quadratic form
        inv_det = 1.0 / determinants
        temp1 = inv_det * (det_22 * innovations[0, :] - det_12 * innovations[1, :])
        temp2 = inv_det * (-det_12 * innovations[0, :] + det_11 * innovations[1, :])
        quadratic_form = temp1 * innovations[0, :] + temp2 * innovations[1, :]
        
        return log_factors - 0.5 * quadratic_form

    def _get_square_2_fast(self, matrices_in: np.ndarray) -> np.ndarray:
        """Fast computation of matrix squares: A @ A for batch of 2x2 matrices."""
        matrices_out = np.zeros_like(matrices_in)
        
        # Extract matrix elements
        a11, a12 = matrices_in[0, 0, :], matrices_in[0, 1, :]
        a21, a22 = matrices_in[1, 0, :], matrices_in[1, 1, :]
        
        # Compute A^2 = A @ A
        matrices_out[0, 0, :] = a11*a11 + a12*a21
        matrices_out[0, 1, :] = a11*a12 + a12*a22
        matrices_out[1, 0, :] = a21*a11 + a22*a21
        matrices_out[1, 1, :] = a21*a12 + a22*a22
        
        return matrices_out

    def _get_weights_unknown(self, log_weights: np.ndarray, old_existence: float,
                            skip_index: int) -> Tuple[np.ndarray, float]:
        """Compute particle weights and updated existence probability."""
        # Zero out weights for skipped measurement
        if skip_index < log_weights.shape[1]:
            log_weights[:, skip_index] = 0
        
        # Sum log weights across measurements
        summed_log_weights = np.sum(log_weights, axis=1)
        
        # Compute existence update
        alive_update = np.mean(np.exp(summed_log_weights))
        if np.isinf(alive_update):
            updated_existence = 1.0
        else:
            alive = old_existence * alive_update
            dead = 1 - old_existence
            updated_existence = alive / (dead + alive)
        
        # Normalize weights
        max_weight = np.max(summed_log_weights)
        weights = np.exp(summed_log_weights - max_weight)
        weights = weights / np.sum(weights)
        
        return weights, updated_existence

    def _update_particles(self, old_particles_kinematic: np.ndarray, old_particles_extent: np.ndarray,
                         old_existence: float, log_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Update particles using importance sampling and resampling."""
        # Sum weights across measurements
        summed_log_weights = np.sum(log_weights, axis=1)
        
        # Update existence probability  
        alive_update = np.mean(np.exp(summed_log_weights))
        if np.isinf(alive_update):
            updated_existence = 1.0
        else:
            alive = old_existence * alive_update
            dead = 1 - old_existence
            updated_existence = alive / (dead + alive)
        
        # Update particles only if target exists
        if updated_existence != 0:
            # Normalize weights
            max_weight = np.max(summed_log_weights)
            weights = np.exp(summed_log_weights - max_weight)
            weights_normalized = weights / np.sum(weights)
            
            # Resample particles
            indexes = self._systematic_resample(weights_normalized)
            updated_particles_kinematic = old_particles_kinematic[:, indexes]
            updated_particles_extent = old_particles_extent[:, :, indexes]
            
            # Add regularization noise to prevent degeneracy
            if self.parameters.regularization_deviation > 0:
                updated_particles_kinematic[:2, :] += (self.parameters.regularization_deviation * 
                                                      np.random.randn(2, self.parameters.num_particles))
        else:
            # Target doesn't exist - return NaN particles
            updated_particles_kinematic = np.full_like(old_particles_kinematic, np.nan)
            updated_particles_extent = np.full_like(old_particles_extent, np.nan)
        
        return updated_particles_kinematic, updated_particles_extent, updated_existence

    def _systematic_resample(self, weights: np.ndarray) -> np.ndarray:
        """Perform systematic resampling of particles."""
        # Cumulative sum of weights
        cumsum = np.cumsum(weights)
        
        # Generate systematic sampling points
        u = np.random.rand() / self.parameters.num_particles
        sampling_points = u + np.arange(self.parameters.num_particles) / self.parameters.num_particles
        
        # Find indices using searchsorted
        indices = np.searchsorted(cumsum, sampling_points)
        
        # Ensure indices are within bounds
        indices = np.minimum(indices, len(weights) - 1)
        
        return indices

    def _iwishart_fast_vector(self, scale_matrix: np.ndarray, df: float, 
                             num_samples: int) -> np.ndarray:
        """Fast vectorized sampling from inverse Wishart distribution."""
        samples = np.zeros((2, 2, num_samples))
        
        for i in range(num_samples):
            samples[:, :, i] = invwishart.rvs(df=df, scale=scale_matrix)
        
        return samples

    def _wishart_fast_vector(self, scale_matrices: np.ndarray, df: float, 
                            num_samples: int) -> np.ndarray:
        """Fast vectorized sampling from Wishart distribution."""
        samples = np.zeros((2, 2, num_samples))
        
        for i in range(num_samples):
            samples[:, :, i] = wishart.rvs(df=df, scale=scale_matrices[:, :, i])
        
        return samples