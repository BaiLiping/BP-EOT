"""
Extended Object Tracking with Elliptical Shapes - Main Algorithm

This module implements the core EOT algorithm using belief propagation
for data association and particle filtering for state estimation.

The algorithm tracks multiple extended objects (targets with spatial extent)
in cluttered environments using elliptical shape models.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from scipy.stats import multivariate_normal, poisson
from scipy.linalg import sqrtm
from dataclasses import dataclass

from eot_common import (
    EOTParameters, systematic_resample, iwishart_fast_vector, 
    wishart_fast_vector
)


@dataclass
class TrackEstimate:
    """Structure to hold track estimates for a single time step."""
    state: np.ndarray      # Kinematic states (4 x num_detections)
    extent: np.ndarray     # Extent matrices (2 x 2 x num_detections)  
    label: np.ndarray      # Track labels (2 x num_detections)


def eot_elliptical_shape(measurements_cell: List[np.ndarray], 
                        parameters: EOTParameters) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extended Object Tracking with Elliptical Shapes.
    
    This is the main EOT algorithm that processes a sequence of measurements
    to estimate the states and extents of multiple extended objects.
    
    The algorithm uses:
    - Belief propagation for data association
    - Particle filtering for state estimation
    - Inverse Wishart distribution for extent modeling
    - Track management for birth/death of objects
    
    Args:
        measurements_cell: List of measurement arrays, one per time step
        parameters: Algorithm parameters
        
    Returns:
        estimated_tracks: Estimated kinematic trajectories (4 x num_steps x num_tracks)
        estimated_extents: Estimated extent trajectories (2 x 2 x num_steps x num_tracks)
    """
    
    # Extract key parameters for readability
    num_particles = parameters.num_particles
    mean_clutter = parameters.mean_clutter
    mean_measurements = parameters.mean_measurements
    scan_time = parameters.scan_time
    detection_threshold = parameters.detection_threshold
    threshold_pruning = parameters.threshold_pruning
    num_outer_iterations = parameters.num_outer_iterations
    mean_births = parameters.mean_births
    prior_velocity_covariance = parameters.prior_velocity_covariance
    surveillance_region = parameters.surveillance_region
    
    # Derived parameters
    area_size = ((surveillance_region[0, 1] - surveillance_region[0, 0]) * 
                 (surveillance_region[1, 1] - surveillance_region[1, 0]))
    measurements_covariance = parameters.measurement_variance * np.eye(2)
    
    prior_extent_1 = parameters.prior_extent_1
    prior_extent_2 = parameters.prior_extent_2
    mean_extent_prior = np.trace(parameters.prior_extent_1 / (parameters.prior_extent_2 - 3))
    total_covariance = mean_extent_prior**2 + measurements_covariance
    
    num_steps = len(measurements_cell)
    constant_factor = area_size * (mean_measurements / mean_clutter)
    uniform_weight = np.log(1.0 / area_size)
    
    # Initialize tracking state
    estimates = [None] * num_steps
    current_labels = np.empty((2, 0))  # Track labels [birth_time; measurement_index]
    current_particles_kinematic = np.empty((4, num_particles, 0))  # Kinematic particles
    current_existences = np.empty(0)  # Existence probabilities
    current_particles_extent = np.empty((2, 2, num_particles, 0))  # Extent particles
    
    
    # ==========================================================================
    # MAIN PROCESSING LOOP - Process each time step
    # ==========================================================================
    
    for step in range(num_steps):
        print(f"Processing time step {step + 1}/{num_steps}")
        
        # Load measurements for current time step
        measurements = measurements_cell[step]
        num_measurements = measurements.shape[1] if measurements.size > 0 else 0
        
        
        # ======================================================================
        # PREDICTION STEP - Propagate existing tracks forward in time
        # ======================================================================
        
        (current_particles_kinematic, current_existences, 
         current_particles_extent) = perform_prediction(
            current_particles_kinematic, current_existences,
            current_particles_extent, scan_time, parameters
        )
        
        # Update existence probabilities (survival and measurement likelihood)
        current_alive = current_existences * np.exp(-mean_measurements)
        current_dead = 1 - current_existences
        current_existences = current_alive / (current_dead + current_alive)
        
        num_targets = current_particles_kinematic.shape[2]
        num_legacy = num_targets  # Number of existing tracks
        
        
        # ======================================================================  
        # BIRTH DETECTION - Find measurements likely to be new targets
        # ======================================================================
        
        if num_measurements > 0:
            new_indexes, measurements = get_promising_new_targets(
                current_particles_kinematic, current_particles_extent,
                current_existences, measurements, parameters
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
        
        
        # ======================================================================
        # INITIALIZE NEW TARGETS - Create particles for potential new targets  
        # ======================================================================
        
        # Initialize belief propagation variables
        new_existences = np.full(num_new, mean_births * np.exp(-mean_measurements) / 
                                (mean_births * np.exp(-mean_measurements) + 1))
        new_particles_kinematic = np.zeros((4, num_particles, num_new))
        new_particles_extent = np.zeros((2, 2, num_particles, num_new))
        new_weights = np.zeros((num_particles, num_new))
        
        # Initialize each new target
        for target in range(num_new):
            if new_indexes[target] < measurements.shape[1]:
                # Proposal distribution centered at measurement location
                proposal_mean = measurements[:, int(new_indexes[target])]
                proposal_covariance = 2 * total_covariance  # Heavy-tailed proposal
                
                # Sample position particles from proposal
                pos_samples = (proposal_mean.reshape(-1, 1) + 
                              sqrtm(proposal_covariance) @ np.random.randn(2, num_particles))
                new_particles_kinematic[:2, :, target] = pos_samples
                
                # Compute importance weights
                for p in range(num_particles):
                    weight = (uniform_weight - 
                             multivariate_normal.logpdf(
                                 new_particles_kinematic[:2, p, target],
                                 proposal_mean, proposal_covariance))
                    new_weights[p, target] = weight
                
                # Sample extent particles from prior
                new_particles_extent[:, :, :, target] = iwishart_fast_vector(
                    prior_extent_1, prior_extent_2, num_particles
                )
        
        # Combine legacy and new targets
        current_existences = np.concatenate([current_existences, new_existences])
        current_existences_extrinsic = np.tile(current_existences, (num_measurements, 1)).T
        
        current_particles_kinematic = np.concatenate([
            current_particles_kinematic, new_particles_kinematic], axis=2)
        current_particles_extent = np.concatenate([
            current_particles_extent, new_particles_extent], axis=3)
        
        # Initialize message passing arrays
        weights_extrinsic = np.full((num_particles, num_measurements, num_legacy), np.nan)
        weights_extrinsic_new = np.full((num_particles, num_measurements, num_new), np.nan)
        
        likelihood_1 = np.zeros((num_particles, num_measurements, num_legacy + num_new))
        likelihood_new_1 = np.full((num_particles, num_measurements, num_new), np.nan)
        
        
        # ======================================================================
        # BELIEF PROPAGATION - Iterate to solve data association
        # ======================================================================
        
        for outer in range(num_outer_iterations):
            
            # Initialize message passing for each measurement
            output_da = [None] * num_measurements
            target_indexes = [None] * num_measurements
            
            # Process measurements in reverse order (important for new target logic)
            for measurement in range(num_measurements - 1, -1, -1):
                
                # Initialize data association input for legacy targets
                input_da = np.ones((2, num_legacy + num_new))
                
                # Process legacy targets
                for target in range(num_legacy):
                    if outer == 0:
                        # First iteration: compute likelihoods
                        extent_matrices = get_square_2_fast(current_particles_extent[:, :, :, target])
                        meas_cov_tiled = np.tile(measurements_covariance[:, :, np.newaxis], (1, 1, num_particles))
                        total_cov = extent_matrices + meas_cov_tiled
                        
                        likelihood_1[:, measurement, target] = (
                            constant_factor * np.exp(get_log_weights_fast(
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
                
                # Process new targets (only those with index >= current measurement)
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
                            
                            extent_matrices = get_square_2_fast(current_particles_extent[:, :, :, target_index])
                            meas_cov_tiled = np.tile(measurements_covariance[:, :, np.newaxis], (1, 1, num_particles))
                            total_cov = extent_matrices + meas_cov_tiled
                            
                            likelihood_new_1[:, measurement, target_idx_in_new] = (
                                constant_factor * np.exp(get_log_weights_fast(
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
                output_da[measurement] = data_association_bp(input_da[:, :target_index])
            
            
            # ==================================================================
            # UPDATE STEP - Update particle weights and existence probabilities
            # ==================================================================
            
            # Update legacy targets
            for target in range(num_legacy):
                weights = np.zeros((num_particles, num_measurements))
                
                for measurement in range(num_measurements):
                    if output_da[measurement] is not None and target < output_da[measurement].shape[0]:
                        current_weights = (1 + 
                                         likelihood_1[:, measurement, target] * 
                                         output_da[measurement][target])
                        weights[:, measurement] = np.log(current_weights)
                
                # Compute extrinsic information or final belief
                if outer != num_outer_iterations - 1:
                    # Not last iteration: compute extrinsic information
                    for measurement in range(num_measurements):
                        (weights_extrinsic[:, measurement, target],
                         current_existences_extrinsic[target, measurement]) = get_weights_unknown(
                            weights, current_existences[target], measurement
                        )
                else:
                    # Last iteration: update particles
                    (current_particles_kinematic[:, :, target],
                     current_particles_extent[:, :, :, target],
                     current_existences[target]) = update_particles(
                        current_particles_kinematic[:, :, target],
                        current_particles_extent[:, :, :, target],
                        current_existences[target], weights, parameters
                    )
            
            # Update new targets
            target_index = num_legacy
            for target_meas in range(num_measurements - 1, -1, -1):
                if target_meas in new_indexes:
                    target_idx_in_new = np.where(new_indexes == target_meas)[0][0]
                    
                    weights = np.zeros((num_particles, num_measurements + 1))
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
                                
                                weights[:, measurement] = np.log(current_weights)
                    
                    # Compute extrinsic information or final belief
                    if outer != num_outer_iterations - 1:
                        # Not last iteration
                        for measurement in range(target_meas + 1):
                            (weights_extrinsic_new[:, measurement, target_idx_in_new],
                             current_existences_extrinsic[target_index, measurement]) = get_weights_unknown(
                                weights, current_existences[target_index], measurement
                            )
                    else:
                        # Last iteration: update particles
                        (current_particles_kinematic[:2, :, target_index],
                         current_particles_extent[:, :, :, target_index],
                         current_existences[target_index]) = update_particles(
                            current_particles_kinematic[:2, :, target_index],
                            current_particles_extent[:, :, :, target_index],
                            current_existences[target_index], weights, parameters
                        )
                        
                        # Sample velocity for new targets
                        current_particles_kinematic[2:4, :, target_index] = (
                            np.random.multivariate_normal([0, 0], prior_velocity_covariance, 
                                                        num_particles).T
                        )
                    
                    target_index += 1
        
        
        # ======================================================================
        # PRUNING - Remove low-probability tracks
        # ======================================================================
        
        num_targets = current_particles_kinematic.shape[2]
        is_redundant = current_existences < threshold_pruning
        
        # Keep only non-redundant tracks
        current_particles_kinematic = current_particles_kinematic[:, :, ~is_redundant]
        current_particles_extent = current_particles_extent[:, :, :, ~is_redundant]
        current_labels = current_labels[:, ~is_redundant]
        current_existences = current_existences[~is_redundant]
        
        
        # ======================================================================
        # ESTIMATION - Extract state estimates for detected targets
        # ======================================================================
        
        num_targets = current_particles_kinematic.shape[2]
        detected_targets = 0
        current_estimate = TrackEstimate(
            state=np.empty((4, 0)),
            extent=np.empty((2, 2, 0)), 
            label=np.empty((2, 0))
        )
        
        for target in range(num_targets):
            if current_existences[target] > detection_threshold:
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
        
        estimates[step] = current_estimate if detected_targets > 0 else None
    
    
    # ==========================================================================
    # TRACK FORMATION - Convert estimates to continuous tracks
    # ==========================================================================
    
    estimated_tracks, estimated_extents = track_formation(estimates, parameters)
    
    return estimated_tracks, estimated_extents


# ==============================================================================
# SUPPORTING FUNCTIONS - Helper functions for the main algorithm
# ==============================================================================

def perform_prediction(old_particles: np.ndarray, old_existences: np.ndarray,
                      old_extents: np.ndarray, scan_time: float, 
                      parameters: EOTParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform prediction step for kinematic states and extent matrices.
    
    This function propagates particles forward in time using:
    - Constant velocity motion model with acceleration noise for kinematics
    - Wishart distribution evolution for extents
    - Survival probability for existence
    """
    if old_particles.shape[2] == 0:
        return old_particles, old_existences, old_extents
    
    num_particles = old_particles.shape[1]
    num_targets = old_particles.shape[2]
    
    driving_noise_variance = parameters.acceleration_deviation**2
    survival_probability = parameters.survival_probability
    degree_freedom_prediction = parameters.degree_freedom_prediction
    
    # Get motion model matrices
    from eot_common import get_transition_matrices
    A, W = get_transition_matrices(scan_time)
    
    # Initialize output arrays
    new_particles = old_particles.copy()
    new_existences = old_existences.copy()
    new_extents = old_extents.copy()
    
    # Predict extent matrices using Wishart evolution
    for target in range(num_targets):
        scale_matrices = old_extents[:, :, :, target] / degree_freedom_prediction
        new_extents[:, :, :, target] = wishart_fast_vector(
            scale_matrices, degree_freedom_prediction, num_particles
        )
    
    # Predict kinematic states and existences
    for target in range(num_targets):
        # Kinematic prediction with process noise
        process_noise = W @ (np.sqrt(driving_noise_variance) * np.random.randn(2, num_particles))
        new_particles[:, :, target] = A @ old_particles[:, :, target] + process_noise
        
        # Existence prediction
        new_existences[target] = survival_probability * old_existences[target]
    
    return new_particles, new_existences, new_extents


def get_promising_new_targets(current_particles_kinematic: np.ndarray, 
                            current_particles_extent: np.ndarray,
                            current_existences: np.ndarray,
                            measurements: np.ndarray,
                            parameters: EOTParameters) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify measurements likely to originate from new targets.
    
    This function uses a simplified birth detection scheme that identifies
    measurements with low association probability to existing targets.
    """
    
    if measurements.size == 0:
        return np.array([]), measurements
    
    num_measurements = measurements.shape[1]
    
    # For simplicity, we'll use a basic approach: 
    # Consider measurements that are far from existing targets as potential new births
    
    if current_particles_kinematic.shape[2] == 0:
        # No existing targets - all measurements are potential births
        return np.arange(num_measurements), measurements
    
    # Compute distance from each measurement to existing target particles
    min_distances = np.full(num_measurements, np.inf)
    
    for measurement_idx in range(num_measurements):
        measurement = measurements[:, measurement_idx]
        
        for target in range(current_particles_kinematic.shape[2]):
            if current_existences[target] > 0.1:  # Only consider probable targets
                target_positions = current_particles_kinematic[:2, :, target]
                distances = np.linalg.norm(
                    target_positions - measurement.reshape(-1, 1), axis=0
                )
                min_distances[measurement_idx] = min(
                    min_distances[measurement_idx], np.min(distances)
                )
    
    # Measurements far from existing targets are potential new births
    birth_threshold = 20.0  # Distance threshold
    promising_indexes = np.where(min_distances > birth_threshold)[0]
    
    # Limit number of new births per time step
    max_new_births = min(5, len(promising_indexes))
    if len(promising_indexes) > max_new_births:
        promising_indexes = promising_indexes[:max_new_births]
    
    return promising_indexes, measurements


def data_association_bp(input_da: np.ndarray) -> np.ndarray:
    """
    Belief propagation for data association.
    
    This implements the BP message passing algorithm for solving the
    data association problem between measurements and targets.
    """
    if input_da.shape[1] == 0:
        return np.array([])
    
    # Perform data association computation
    input_ratios = input_da[1, :] / input_da[0, :]
    sum_input = 1 + np.sum(input_ratios)
    output_da = 1.0 / (sum_input - input_ratios)
    
    # Handle NaN cases with hard association
    if np.any(np.isnan(output_da)):
        output_da = np.zeros_like(output_da)
        max_idx = np.argmax(input_ratios)
        output_da[max_idx] = 1.0
    
    return output_da


def get_log_weights_fast(measurement: np.ndarray, particles_kinematic: np.ndarray,
                        particles_extent: np.ndarray) -> np.ndarray:
    """
    Fast computation of log-likelihood weights for measurement-to-particle association.
    
    This computes the log-probability of a measurement given particle states
    and extent matrices using efficient vectorized operations.
    """
    num_particles = particles_extent.shape[2]
    
    # Compute determinants efficiently
    det_11 = particles_extent[0, 0, :]
    det_22 = particles_extent[1, 1, :]  
    det_12 = particles_extent[0, 1, :]
    determinants = det_11 * det_22 - det_12**2
    
    # Normalization factors
    log_factors = np.log(1.0 / (2 * np.pi * np.sqrt(determinants)))
    
    # Innovation vectors
    innovations = measurement.reshape(-1, 1) - particles_kinematic[:2, :]  # 2 x num_particles
    
    # Efficient computation of quadratic form: innovation^T * inv(cov) * innovation
    inv_det = 1.0 / determinants
    
    # Compute inv(cov) * innovation efficiently
    temp1 = inv_det * (det_22 * innovations[0, :] - det_12 * innovations[1, :])
    temp2 = inv_det * (-det_12 * innovations[0, :] + det_11 * innovations[1, :])
    
    # Quadratic form
    quadratic_form = temp1 * innovations[0, :] + temp2 * innovations[1, :]
    
    log_weights = log_factors - 0.5 * quadratic_form
    
    return log_weights


def get_square_2_fast(matrices_in: np.ndarray) -> np.ndarray:
    """
    Fast computation of matrix squares: A @ A for a batch of 2x2 matrices.
    
    This efficiently computes A^2 for multiple matrices simultaneously.
    """
    matrices_out = np.zeros_like(matrices_in)
    
    # Extract matrix elements
    a11, a12 = matrices_in[0, 0, :], matrices_in[0, 1, :]
    a21, a22 = matrices_in[1, 0, :], matrices_in[1, 1, :]
    
    # Compute A^2 = A @ A
    matrices_out[0, 0, :] = a11*a11 + a12*a21  # (1,1) element
    matrices_out[0, 1, :] = a11*a12 + a12*a22  # (1,2) element
    matrices_out[1, 0, :] = a21*a11 + a22*a21  # (2,1) element  
    matrices_out[1, 1, :] = a21*a12 + a22*a22  # (2,2) element
    
    return matrices_out


def get_weights_unknown(log_weights: np.ndarray, old_existence: float,
                       skip_index: int) -> Tuple[np.ndarray, float]:
    """
    Compute particle weights and updated existence probability.
    
    This function computes importance weights for particles and updates
    the target existence probability based on measurement evidence.
    """
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


def update_particles(old_particles_kinematic: np.ndarray, old_particles_extent: np.ndarray,
                    old_existence: float, log_weights: np.ndarray,
                    parameters: EOTParameters) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Update particles using importance sampling and resampling.
    
    This function:
    1. Updates existence probability based on measurement evidence
    2. Resamples particles based on importance weights
    3. Applies regularization to prevent particle degeneracy
    """
    num_particles = parameters.num_particles
    regularization_deviation = parameters.regularization_deviation
    
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
        indexes = systematic_resample(weights_normalized, num_particles)
        updated_particles_kinematic = old_particles_kinematic[:, indexes]
        updated_particles_extent = old_particles_extent[:, :, indexes]
        
        # Add regularization noise to prevent degeneracy
        if regularization_deviation > 0:
            updated_particles_kinematic[:2, :] += (regularization_deviation * 
                                                  np.random.randn(2, num_particles))
    else:
        # Target doesn't exist - return NaN particles
        updated_particles_kinematic = np.full_like(old_particles_kinematic, np.nan)
        updated_particles_extent = np.full_like(old_particles_extent, np.nan)
    
    return updated_particles_kinematic, updated_particles_extent, updated_existence


def track_formation(estimates: List[TrackEstimate], 
                   parameters: EOTParameters) -> Tuple[np.ndarray, np.ndarray]:
    """
    Form continuous tracks from frame-by-frame estimates.
    
    This function connects estimates across time steps using track labels
    and filters out tracks that are too short.
    """
    num_steps = len(estimates)
    minimum_track_length = parameters.minimum_track_length
    
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
    
    # Convert back to array format
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
        if track_length >= minimum_track_length:
            valid_tracks.append(track)
    
    if len(valid_tracks) > 0:
        tracks = tracks[:, :, valid_tracks]
        extents = extents[:, :, :, valid_tracks]
    else:
        tracks = np.empty((4, num_steps, 0))
        extents = np.empty((2, 2, num_steps, 0))
    
    return tracks, extents