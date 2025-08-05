import numpy as np
from scipy.stats import wishart, multivariate_normal, chi2
from typing import List, Dict, Any, Tuple, Optional
import copy
from ExtendedObjectFilter import ExtendedObjectFilter

class ExtendedObjectFilterFixed(ExtendedObjectFilter):
    """
    Fixed Extended Object Filter with proper new target detection.
    Inherits from the original filter and overrides the simplified methods.
    """
    
    def get_promising_new_targets(self, curr_kin: np.ndarray, curr_ext: np.ndarray, 
                                 curr_exist: np.ndarray, measurements: np.ndarray,
                                 params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify promising measurements as new target initiations using clustering.
        
        Args:
            curr_kin: Current kinematic particles (state_dim, num_particles, num_targets)
            curr_ext: Current extent particles (2, 2, num_particles, num_targets)
            curr_exist: Current existence probabilities (num_targets,)
            measurements: Current measurements (2, num_measurements)
            params: Tracking parameters
            
        Returns:
            central_indexes: Indices of measurements to initialize as new targets
            reordered_measurements: Reordered measurements
        """
        if measurements.shape[1] == 0:
            return np.array([]), measurements
            
        num_measurements = measurements.shape[1]
        num_particles = curr_kin.shape[1] if curr_kin.shape[2] > 0 else params['numParticles']
        num_targets = curr_kin.shape[2]
        
        # Measurement noise covariance
        meas_cov = params['measurementVariance'] * np.eye(2)
        
        # Compute surveillance area
        surveillance_region = params['surveillanceRegion']
        area_size = ((surveillance_region[1, 0] - surveillance_region[0, 0]) * 
                    (surveillance_region[1, 1] - surveillance_region[0, 1]))
        
        # Clutter-adjusted constant factor
        mean_meas = params['meanMeasurements']
        mean_clutter = params['meanClutter']
        const_factor = area_size * (mean_meas / mean_clutter)
        
        probabilities_new = np.ones(num_measurements)
        
        # For each measurement, compute probability it's from a new target
        for m in range(num_measurements):
            if num_targets > 0:
                # Prepare data association inputs
                input_da = np.ones((2, num_targets))
                likelihoods = np.zeros((num_particles, num_targets))
                
                for t in range(num_targets):
                    # Build per-particle covariance including extent
                    extent_cov = self.get_square2_fast(curr_ext[:, :, :, t])
                    cov_stack = extent_cov + np.tile(meas_cov[:, :, np.newaxis], (1, 1, num_particles))
                    
                    # Compute log-likelihoods for this target
                    log_w = self.get_log_weights_fast(measurements[:, m], curr_kin[0:2, :, t], cov_stack)
                    likelihoods[:, t] = const_factor * np.exp(log_w)
                    
                    # Expected likelihood (mean over particles)
                    input_da[1, t] = np.mean(likelihoods[:, t])
                    
                    # Weight by existence probability
                    input_da[:, t] = curr_exist[t] * input_da[:, t] + (1 - curr_exist[t]) * np.array([1, 0])
                
                # Compute probability that this measurement is from a new target
                ratio = input_da[1, :] / input_da[0, :]
                sum_da = 1 + np.sum(ratio)
                probabilities_new[m] = 1.0 / sum_da
            
        # Cluster measurements and select central ones
        central_indexes, reorder_idx = self.get_central_reordered(measurements, probabilities_new, params)
        reordered_measurements = measurements[:, reorder_idx]
        
        return central_indexes, reordered_measurements
    
    def get_central_reordered(self, measurements: np.ndarray, probs_new: np.ndarray, 
                             params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster 'free' measurements and select central ones.
        
        Args:
            measurements: Measurements (2, N)
            probs_new: Probability each measurement is from new target (N,)
            params: Tracking parameters
            
        Returns:
            central_indexes: Indices of central measurements for new targets
            indexes_reordered: Reordered measurement indices
        """
        # Clustering parameters
        free_th = 0.85
        cluster_th = 0.9
        min_cluster_elems = 2
        
        # Threshold for considering a measurement 'free' (new target candidate)
        free_mask = probs_new >= free_th
        
        # Separate free and assigned measurement indices
        all_idx = np.arange(measurements.shape[1])
        free_idx = all_idx[free_mask]
        assigned_idx = all_idx[~free_mask]
        
        free_meas = measurements[:, free_mask]
        
        # Covariance used for clustering (birth extent + measurement noise)
        mean_extent_birth = np.trace(params['priorExtent1'] / (params['priorExtent2'] - 3)) / 2
        birth_cov = params['measurementVariance'] * np.eye(2) + mean_extent_birth * np.eye(2)
        
        # Perform clustering
        clusters = self.get_clusters(free_meas, birth_cov, cluster_th)
        
        # Determine clusters with enough elements
        valid_clusters = [c for c in clusters if len(c) >= min_cluster_elems]
        unused = [free_idx[i] for i in range(len(free_idx)) 
                 if not any(i in cluster for cluster in valid_clusters)]
        
        # For each valid cluster, find the central measurement
        central_idxs = []
        ordered_free = []
        
        for cluster in valid_clusters:
            cluster_global_idx = [free_idx[i] for i in cluster]
            pts = free_meas[:, cluster]
            
            if len(cluster) > 1:
                # Compute pairwise Mahalanobis distances
                D = np.zeros((len(cluster), len(cluster)))
                inv_cov = np.linalg.inv(birth_cov)
                
                for i in range(len(cluster)):
                    for j in range(i+1, len(cluster)):
                        diff = pts[:, i] - pts[:, j]
                        D[i, j] = np.sqrt(diff.T @ inv_cov @ diff)
                        D[j, i] = D[i, j]
                
                # Select the point with the smallest average distance (most central)
                sums = D.sum(axis=1)
                center_local_idx = np.argmin(sums)
                center_idx = cluster_global_idx[center_local_idx]
            else:
                center_idx = cluster_global_idx[0]
                
            central_idxs.append(center_idx)
            # Order remaining cluster points after central
            ordered_free.extend([idx for idx in cluster_global_idx if idx != center_idx])
        
        # Build final reorder list
        idx_reordered = unused + central_idxs + ordered_free + assigned_idx.tolist()
        
        return np.array(central_idxs), np.array(idx_reordered)
    
    def get_clusters(self, measurements: np.ndarray, cov: np.ndarray, 
                    threshold_prob: float) -> List[List[int]]:
        """
        Agglomerative clustering based on Mahalanobis distance.
        
        Args:
            measurements: Measurements to cluster (2, N)
            cov: Covariance for distance calculation (2, 2)
            threshold_prob: Probability threshold for clustering
            
        Returns:
            clusters: List of clusters, each containing measurement indices
        """
        num = measurements.shape[1]
        if num == 0:
            return []
        
        # Distance threshold from chi-square inverse CDF
        thresh_dist = np.sqrt(chi2.ppf(threshold_prob, df=2))
        
        # Compute pairwise Mahalanobis distances
        dist_mat = np.zeros((num, num))
        inv_cov = np.linalg.inv(cov)
        
        for i in range(num):
            for j in range(i+1, num):
                diff = measurements[:, i] - measurements[:, j]
                d = np.sqrt(diff.T @ inv_cov @ diff)
                dist_mat[i, j] = d
                dist_mat[j, i] = d
        
        # Build clusters by flood-fill
        unvisited = set(range(num))
        clusters = []
        
        while unvisited:
            seed = unvisited.pop()
            cluster = {seed}
            stack = [seed]
            
            while stack:
                idx = stack.pop()
                neighbors = [j for j in range(num)
                           if j in unvisited and dist_mat[idx, j] < thresh_dist]
                for nb in neighbors:
                    unvisited.remove(nb)
                    cluster.add(nb)
                    stack.append(nb)
                    
            clusters.append(sorted(cluster))
            
        return clusters
    
    def eot_elliptical_shape(self, measurements_cell: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Extended object tracking with elliptical shape using belief propagation.
        This is a fixed version that properly implements new target detection.
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
        
        # Calculate constants
        area_size = ((surveillance_region[1, 0] - surveillance_region[0, 0]) * 
                    (surveillance_region[1, 1] - surveillance_region[0, 1]))
        
        if area_size <= 0:
            raise ValueError(f"Invalid surveillance region area: {area_size}, region: {surveillance_region}")
            
        measurements_covariance = measurement_variance * np.eye(2)
        mean_extent_prior = np.trace(prior_extent1 / (prior_extent2 - 3)) / 2
        total_covariance = mean_extent_prior * np.eye(2) + measurements_covariance
        
        num_steps = len(measurements_cell)
        constant_factor = area_size * (mean_measurements / mean_clutter)
        uniform_weight = np.log(1.0 / area_size)
        
        # Initialize tracking variables
        estimates = []
        current_labels = np.zeros((2, 0))
        current_particles_kinematic = np.zeros((4, num_particles, 0))
        current_existences = np.zeros(0)
        current_particles_extent = np.zeros((2, 2, num_particles, 0))
        
        # Main tracking loop
        for step in range(num_steps):
            print(f"Processing step {step + 1}")
            
            # Load current measurements
            measurements = measurements_cell[step]
            num_measurements = measurements.shape[1] if measurements.ndim > 1 else 0
            
            if num_measurements == 0:
                estimates.append([])
                continue
            
            # Perform prediction step
            if current_particles_kinematic.shape[2] > 0:
                (current_particles_kinematic, current_existences, 
                 current_particles_extent) = self.perform_prediction(
                    current_particles_kinematic, current_existences, 
                    current_particles_extent, scan_time, self.parameters
                )
                
            # Update existence probabilities
            current_alive = current_existences * np.exp(-mean_measurements)
            current_dead = 1 - current_existences
            current_existences = current_alive / (current_dead + current_alive)
            num_targets = current_particles_kinematic.shape[2]
            num_legacy = num_targets
            
            # Get promising new targets using proper clustering
            new_indexes, measurements = self.get_promising_new_targets(
                current_particles_kinematic, current_particles_extent,
                current_existences, measurements, self.parameters
            )
            
            num_new = len(new_indexes)
            
            if num_new > 0:
                new_labels = np.array([[step + 1] * num_new, new_indexes])
                current_labels = np.concatenate([current_labels, new_labels], axis=1)
                
                # Initialize new targets
                new_existences = np.full(num_new, mean_births * np.exp(-mean_measurements) / 
                                       (mean_births * np.exp(-mean_measurements) + 1))
                new_particles_kinematic = np.zeros((4, num_particles, num_new))
                new_particles_extent = np.zeros((2, 2, num_particles, num_new))
                
                for target in range(num_new):
                    proposal_mean = measurements[:, new_indexes[target]]
                    proposal_covariance = 2 * total_covariance
                    
                    # Sample positions
                    new_particles_kinematic[0:2, :, target] = (
                        proposal_mean[:, np.newaxis] + 
                        np.linalg.cholesky(proposal_covariance) @ np.random.randn(2, num_particles)
                    )
                    
                    # Sample velocities
                    new_particles_kinematic[2:4, :, target] = (
                        np.random.multivariate_normal([0, 0], prior_velocity_covariance, 
                                                    num_particles).T
                    )
                    
                    # Sample extents
                    new_particles_extent[:, :, :, target] = self.iwishrnd_fast_vector(
                        prior_extent1, prior_extent2, num_particles
                    )
                
                # Append new targets
                current_existences = np.concatenate([current_existences, new_existences])
                current_particles_kinematic = np.concatenate(
                    [current_particles_kinematic, new_particles_kinematic], axis=2
                )
                current_particles_extent = np.concatenate(
                    [current_particles_extent, new_particles_extent], axis=3
                )
            
            # Simplified belief propagation update (for all targets)
            num_targets = current_particles_kinematic.shape[2]
            
            for target in range(num_targets):
                # Compute likelihoods for all measurements
                log_weights = np.zeros((num_particles, num_measurements))
                
                for m in range(num_measurements):
                    extent_cov = self.get_square2_fast(current_particles_extent[:, :, :, target])
                    cov_stack = extent_cov + np.tile(measurements_covariance[:, :, np.newaxis], 
                                                    (1, 1, num_particles))
                    
                    log_weights[:, m] = self.get_log_weights_fast(
                        measurements[:, m], 
                        current_particles_kinematic[0:2, :, target], 
                        cov_stack
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