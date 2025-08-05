#!/usr/bin/env python3
"""
Extended Object Tracking implementation based purely on the mathematical formulation 
from the paper by Meyer & Williams (2021).

"Scalable Detection and Tracking of Geometric Extended Objects"
IEEE Trans. Signal Process., vol. 69, pp. 6283–6298, Oct. 2021.

This implementation follows the factor graph and sum-product algorithm 
as described in the paper, without referencing any existing code.
"""

import numpy as np
from scipy.stats import multivariate_normal, wishart, poisson
from scipy.linalg import sqrtm, inv
from typing import List, Dict, Any, Tuple, Optional
import copy

class PotentialObject:
    """
    Represents a potential object with kinematic state, extent state, and existence.
    
    State representation from paper:
    - Kinematic state: x_{k,n} = [p_{k,n}^T, m_{k,n}^T]^T (position + velocity)
    - Extent state: E_{k,n} (positive semidefinite matrix)  
    - Existence: r_{k,n} ∈ {0,1}
    """
    
    def __init__(self, kinematic_dim: int = 4, spatial_dim: int = 2, num_particles: int = 1000):
        """
        Initialize potential object.
        
        Args:
            kinematic_dim: Dimension of kinematic state (position + velocity)
            spatial_dim: Spatial dimension (2D tracking)
            num_particles: Number of particles for particle filtering
        """
        self.kinematic_dim = kinematic_dim
        self.spatial_dim = spatial_dim
        self.num_particles = num_particles
        
        # Particle representation of kinematic state
        self.kinematic_particles = np.zeros((kinematic_dim, num_particles))
        
        # Particle representation of extent state (2x2 matrices for each particle)
        self.extent_particles = np.zeros((spatial_dim, spatial_dim, num_particles))
        
        # Existence probability
        self.existence_prob = 0.0
        
        # Particle weights
        self.weights = np.ones(num_particles) / num_particles
        
        # Label for tracking
        self.label = None

class MeasurementOrientedEOT:
    """
    Extended Object Tracking using measurement-oriented data association
    and belief propagation on factor graphs.
    
    Based on the mathematical formulation in Meyer & Williams (2021).
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize the Extended Object Tracker.
        
        Args:
            params: Dictionary containing tracking parameters
        """
        self.params = params
        
        # Extract key parameters from paper
        self.num_particles = params.get('num_particles', 1000)
        self.survival_prob = params.get('survival_probability', 0.99)  # p_s
        self.num_iterations = params.get('num_outer_iterations', 3)    # P
        self.detection_threshold = params.get('detection_threshold', 0.5)
        self.pruning_threshold = params.get('pruning_threshold', 1e-3)
        
        # Measurement model parameters
        self.measurement_noise_cov = params.get('measurement_variance', 1.0) * np.eye(2)
        self.mu_fa = params.get('mean_clutter', 10.0)  # False alarm rate
        self.surveillance_region = params.get('surveillance_region', 
                                            np.array([[-200, -200], [200, 200]]))
        
        # Calculate surveillance area
        self.surveillance_area = ((self.surveillance_region[1, 0] - self.surveillance_region[0, 0]) * 
                                (self.surveillance_region[1, 1] - self.surveillance_region[0, 1]))
        
        # False alarm PDF (uniform over surveillance region)
        self.f_fa = 1.0 / self.surveillance_area
        
        # New object birth parameters
        self.mu_n = params.get('mean_births', 0.01)  # Birth rate
        self.prior_extent1 = params.get('prior_extent1', np.array([[9, 0], [0, 9]]))
        self.prior_extent2 = params.get('prior_extent2', 100)
        self.prior_velocity_cov = params.get('prior_velocity_covariance', 
                                           np.diag([100, 100]))
        
        # Process noise parameters
        self.acceleration_std = params.get('acceleration_deviation', 1.0)
        self.scan_time = params.get('scan_time', 0.2)
        
        # Current legacy objects
        self.legacy_objects: List[PotentialObject] = []
        
    def measurement_rate_function(self, kinematic_state: np.ndarray, 
                                extent_state: np.ndarray) -> float:
        """
        Compute expected number of measurements μ_m(x,e) from an object.
        
        From paper: Poisson distributed with mean related to object extent.
        Simple model: μ_m = ρ * |E| where ρ is measurement density.
        
        Args:
            kinematic_state: Kinematic state vector
            extent_state: Extent matrix (2x2)
            
        Returns:
            Expected measurement rate
        """
        # Simple model: rate proportional to extent determinant (area)
        det_extent = np.linalg.det(extent_state)
        if det_extent <= 0:
            return 1e-6
        
        measurement_density = self.params.get('mean_measurements', 8) / 9.0  # Default extent area
        return measurement_density * np.sqrt(det_extent)
    
    def measurement_likelihood(self, measurement: np.ndarray, 
                             kinematic_state: np.ndarray,
                             extent_state: np.ndarray) -> float:
        """
        Compute likelihood f(z_l | x_k, e_k) of measurement given object state.
        
        From paper equation (31): Gaussian around object position with extent covariance.
        For additive Gaussian model: f(z_l | x_k, e_k) = N(z_l; p_k, E_k^2 + Σ_u)
        
        Args:
            measurement: Measurement vector (2D)
            kinematic_state: Kinematic state (position + velocity)
            extent_state: Extent matrix (2x2)
            
        Returns:
            Measurement likelihood
        """
        position = kinematic_state[:2]  # Extract position part
        
        # Covariance = extent^2 + measurement noise
        try:
            extent_squared = extent_state @ extent_state
            covariance = extent_squared + self.measurement_noise_cov
            
            # Compute multivariate normal likelihood
            likelihood = multivariate_normal.pdf(measurement, position, covariance)
            return max(likelihood, 1e-10)  # Numerical stability
            
        except:
            return 1e-10
    
    def predict_legacy_objects(self, dt: float):
        """
        Prediction step for legacy potential objects.
        
        From paper equation (9): 
        α̲(x̲_k, ē_k, r̲_k) = ∫∫ q̲(x̲_k, ē_k, r̲_k | x^-_k, e^-_k, r^-_k) × f̃(x^-_k, e^-_k, r^-_k) dx^-_k de^-_k
        
        Args:
            dt: Time step
        """
        for obj in self.legacy_objects:
            # Predict kinematic state using constant velocity model
            # State transition: [x, y, vx, vy] -> [x+vx*dt, y+vy*dt, vx, vy]
            F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt], 
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
            
            # Process noise covariance
            q = self.acceleration_std**2
            Q = np.array([[dt**4/4, 0, dt**3/2, 0],
                         [0, dt**4/4, 0, dt**3/2],
                         [dt**3/2, 0, dt**2, 0],
                         [0, dt**3/2, 0, dt**2]]) * q
            
            # Predict each particle
            for i in range(obj.num_particles):
                # Kinematic prediction
                obj.kinematic_particles[:, i] = F @ obj.kinematic_particles[:, i]
                obj.kinematic_particles[:, i] += np.random.multivariate_normal(
                    np.zeros(4), Q
                )
                
                # Extent prediction using Wishart distribution
                # Simple model: add small amount of uncertainty
                extent_noise = 0.1 * np.eye(2)
                obj.extent_particles[:, :, i] = (
                    obj.extent_particles[:, :, i] + extent_noise
                )
                
                # Ensure positive semidefinite
                obj.extent_particles[:, :, i] = self._make_positive_semidefinite(
                    obj.extent_particles[:, :, i]
                )
            
            # Update existence probability using survival model
            # From paper equation (11): survival probability p_s
            obj.existence_prob = self.survival_prob * obj.existence_prob
            
    def _make_positive_semidefinite(self, matrix: np.ndarray) -> np.ndarray:
        """Make matrix positive semidefinite via eigenvalue decomposition."""
        try:
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, 1e-6)  # Ensure positive
            return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        except:
            return np.eye(matrix.shape[0]) * 1e-3
    
    def initialize_new_objects(self, measurements: np.ndarray) -> List[PotentialObject]:
        """
        Initialize new potential objects from measurements.
        
        From paper: New objects modeled by Poisson birth process.
        Each measurement is a potential new object with prior distribution.
        
        Args:
            measurements: Array of measurements (2, M)
            
        Returns:
            List of new potential objects
        """
        if measurements.shape[1] == 0:
            return []
        
        new_objects = []
        num_measurements = measurements.shape[1]
        
        # Create new potential object for each measurement
        for m in range(num_measurements):
            obj = PotentialObject(num_particles=self.num_particles)
            
            # Initialize particles around measurement location
            measurement_pos = measurements[:, m]
            
            # Prior for new objects from paper: birth distribution f_n(x, e)
            for i in range(obj.num_particles):
                # Position: Gaussian around measurement
                pos_noise_cov = 4.0 * self.measurement_noise_cov  # Larger uncertainty
                obj.kinematic_particles[:2, i] = (
                    measurement_pos + 
                    np.random.multivariate_normal(np.zeros(2), pos_noise_cov)
                )
                
                # Velocity: Zero mean with prior covariance
                obj.kinematic_particles[2:, i] = np.random.multivariate_normal(
                    np.zeros(2), self.prior_velocity_cov
                )
                
                # Extent: Inverse Wishart prior
                obj.extent_particles[:, :, i] = self._sample_inverse_wishart(
                    self.prior_extent1, self.prior_extent2
                )
            
            # Initial existence probability for new objects
            # From paper: related to birth rate and measurement rate
            mu_m_expected = 8.0  # Expected measurement rate
            obj.existence_prob = (
                self.mu_n * np.exp(-mu_m_expected) / 
                (self.mu_n * np.exp(-mu_m_expected) + 1)
            )
            
            obj.label = f"new_{m}"
            new_objects.append(obj)
            
        return new_objects
    
    def _sample_inverse_wishart(self, scale_matrix: np.ndarray, 
                               degrees_freedom: float) -> np.ndarray:
        """
        Sample from inverse Wishart distribution.
        
        Args:
            scale_matrix: Scale matrix
            degrees_freedom: Degrees of freedom
            
        Returns:
            Sample from inverse Wishart
        """
        try:
            # Sample from Wishart then invert
            wishart_sample = wishart.rvs(degrees_freedom, scale_matrix)
            return inv(wishart_sample)
        except:
            # Fallback to identity if sampling fails
            return np.eye(scale_matrix.shape[0])
    
    def compute_measurement_evaluation_messages(self, measurements: np.ndarray,
                                              all_objects: List[PotentialObject],
                                              iteration: int) -> Dict[str, np.ndarray]:
        """
        Compute β messages from measurement evaluation.
        
        From paper equation (15): Messages from pseudo-likelihood factors h_k to association variables b_l.
        β_{kl}^(p)(b_l = k) = (1/μ_fa f_fa(z_l) α_{kl}^(p)) ∫∫ μ_m(x_k,e_k) f(z_l|x_k,e_k) α_{kl}^(p)(x_k,e_k,r_k=1) dx_k de_k
        
        Args:
            measurements: Measurement array (2, M)
            all_objects: List of all potential objects (legacy + new)
            iteration: Current iteration number
            
        Returns:
            Dictionary of β messages
        """
        num_measurements = measurements.shape[1]
        num_objects = len(all_objects)
        
        beta_messages = {}
        
        for k, obj in enumerate(all_objects):
            for l in range(num_measurements):
                measurement = measurements[:, l]
                
                # Compute expected likelihood over particles
                likelihood_sum = 0.0
                weight_sum = 0.0
                
                for i in range(obj.num_particles):
                    kin_state = obj.kinematic_particles[:, i]
                    ext_state = obj.extent_particles[:, :, i] 
                    weight = obj.weights[i]
                    
                    # Measurement rate μ_m(x,e)
                    mu_m = self.measurement_rate_function(kin_state, ext_state)
                    
                    # Measurement likelihood f(z_l | x_k, e_k)
                    likelihood = self.measurement_likelihood(measurement, kin_state, ext_state)
                    
                    likelihood_sum += weight * mu_m * likelihood
                    weight_sum += weight
                
                # Normalize by particle weights
                if weight_sum > 0:
                    expected_likelihood = likelihood_sum / weight_sum
                else:
                    expected_likelihood = 1e-10
                
                # β message from paper equation (15)
                beta_kl = expected_likelihood / (self.mu_fa * self.f_fa)
                beta_messages[f"beta_{k}_{l}"] = beta_kl
                
        return beta_messages
    
    def compute_data_association_messages(self, beta_messages: Dict[str, np.ndarray],
                                        num_measurements: int,
                                        num_objects: int) -> Dict[str, np.ndarray]:
        """
        Compute ν messages for data association.
        
        From paper equation (18): Messages from association variables b_l to pseudo-likelihood factors.
        Product of all β messages except the one being computed.
        
        Args:
            beta_messages: β messages from measurement evaluation
            num_measurements: Number of measurements  
            num_objects: Number of objects
            
        Returns:
            Dictionary of ν messages
        """
        nu_messages = {}
        
        for l in range(num_measurements):
            for k in range(num_objects):
                # ν message is product of all other β messages for this measurement
                nu_product = 1.0
                
                for k_prime in range(num_objects):
                    if k_prime != k:
                        beta_key = f"beta_{k_prime}_{l}"
                        if beta_key in beta_messages:
                            nu_product *= beta_messages[beta_key]
                
                # Add false alarm component (always 1.0 in normalized form)
                nu_product *= 1.0  # False alarm contribution
                
                nu_messages[f"nu_{l}_{k}"] = nu_product
                
        return nu_messages
    
    def compute_measurement_update_messages(self, measurements: np.ndarray,
                                          nu_messages: Dict[str, np.ndarray],
                                          all_objects: List[PotentialObject]) -> Dict[str, Any]:
        """
        Compute γ messages for measurement update.
        
        From paper equation (20): Messages from pseudo-likelihood factors to object states.
        γ_{lk}^(p)(x_k, e_k, r_k = 1) = [μ_m(x_k,e_k) f(z_l|x_k,e_k) / (μ_fa f_fa(z_l))] + ξ_{kl}^(p)
        
        Args:
            measurements: Measurement array
            nu_messages: ν messages from data association
            all_objects: List of all potential objects
            
        Returns:
            Dictionary of γ messages and updated beliefs
        """
        num_measurements = measurements.shape[1]
        gamma_messages = {}
        
        for k, obj in enumerate(all_objects):
            # Accumulate log weights for this object
            log_weights = np.zeros(obj.num_particles)
            
            for l in range(num_measurements):
                measurement = measurements[:, l]
                
                # Get ν message for this measurement-object pair
                nu_key = f"nu_{l}_{k}"
                nu_value = nu_messages.get(nu_key, 1.0)
                
                # Compute particle-specific contributions
                for i in range(obj.num_particles):
                    kin_state = obj.kinematic_particles[:, i]
                    ext_state = obj.extent_particles[:, :, i]
                    
                    # Measurement rate and likelihood
                    mu_m = self.measurement_rate_function(kin_state, ext_state)
                    likelihood = self.measurement_likelihood(measurement, kin_state, ext_state)
                    
                    # γ message contribution (in log space for stability)
                    gamma_contrib = (mu_m * likelihood) / (self.mu_fa * self.f_fa)
                    log_weights[i] += np.log(gamma_contrib * nu_value + 1.0)
            
            # Update object weights and existence probability
            obj.weights = np.exp(log_weights - np.max(log_weights))  # Numerical stability
            obj.weights /= np.sum(obj.weights)  # Normalize
            
            # Update existence probability
            mean_weight = np.mean(obj.weights)
            obj.existence_prob = (obj.existence_prob * mean_weight) / (
                obj.existence_prob * mean_weight + (1 - obj.existence_prob)
            )
            
            gamma_messages[f"gamma_{k}"] = log_weights
            
        return gamma_messages
    
    def run_belief_propagation(self, measurements: np.ndarray,
                             all_objects: List[PotentialObject]) -> List[PotentialObject]:
        """
        Run iterative belief propagation (sum-product algorithm).
        
        From paper Section III-B: Iterative message passing with P iterations.
        Each iteration: measurement evaluation → data association → measurement update
        
        Args:
            measurements: Current measurements (2, M)
            all_objects: All potential objects (legacy + new)
            
        Returns:
            Updated objects after belief propagation
        """
        num_measurements = measurements.shape[1]
        num_objects = len(all_objects)
        
        if num_measurements == 0 or num_objects == 0:
            return all_objects
        
        # Iterative message passing
        for iteration in range(self.num_iterations):
            # Step 1: Measurement Evaluation (β messages)
            beta_messages = self.compute_measurement_evaluation_messages(
                measurements, all_objects, iteration
            )
            
            # Step 2: Data Association (ν messages)  
            nu_messages = self.compute_data_association_messages(
                beta_messages, num_measurements, num_objects
            )
            
            # Step 3: Measurement Update (γ messages)
            gamma_messages = self.compute_measurement_update_messages(
                measurements, nu_messages, all_objects
            )
            
            # Resample particles if needed (systematic resampling)
            for obj in all_objects:
                if self._effective_sample_size(obj.weights) < obj.num_particles / 2:
                    self._resample_particles(obj)
                    
        return all_objects
    
    def _effective_sample_size(self, weights: np.ndarray) -> float:
        """Compute effective sample size."""
        return 1.0 / np.sum(weights**2)
    
    def _resample_particles(self, obj: PotentialObject):
        """Systematic resampling of particles."""
        weights = obj.weights
        num_particles = len(weights)
        
        # Systematic resampling
        positions = (np.random.random() + np.arange(num_particles)) / num_particles
        cumsum = np.cumsum(weights)
        
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, num_particles - 1)
        
        # Resample particles
        obj.kinematic_particles = obj.kinematic_particles[:, indices]
        obj.extent_particles = obj.extent_particles[:, :, indices]
        obj.weights = np.ones(num_particles) / num_particles
    
    def prune_objects(self, all_objects: List[PotentialObject]) -> List[PotentialObject]:
        """
        Prune objects with low existence probability.
        
        Args:
            all_objects: List of all potential objects
            
        Returns:
            Pruned list of objects
        """
        pruned_objects = []
        
        for obj in all_objects:
            if obj.existence_prob >= self.pruning_threshold:
                pruned_objects.append(obj)
                
        return pruned_objects
    
    def detect_objects(self, all_objects: List[PotentialObject]) -> List[Dict[str, Any]]:
        """
        Detect objects by thresholding existence probabilities.
        Compute MMSE estimates for detected objects.
        
        From paper equation (3): Detection by comparing p(r_{k,n} = 1|z_{1:n}) with threshold.
        From paper equation (2): MMSE estimates.
        
        Args:
            all_objects: List of potential objects
            
        Returns:
            List of detected object estimates
        """
        detections = []
        
        for k, obj in enumerate(all_objects):
            if obj.existence_prob > self.detection_threshold:
                # MMSE estimate of kinematic state
                kinematic_estimate = np.average(
                    obj.kinematic_particles, axis=1, weights=obj.weights
                )
                
                # MMSE estimate of extent state
                extent_estimate = np.average(
                    obj.extent_particles, axis=2, weights=obj.weights
                )
                
                detection = {
                    'kinematic_state': kinematic_estimate,
                    'extent_state': extent_estimate,
                    'existence_prob': obj.existence_prob,
                    'label': obj.label or f"object_{k}",
                    'object_id': k
                }
                
                detections.append(detection)
                
        return detections
    
    def process_time_step(self, measurements: np.ndarray, time_step: int) -> List[Dict[str, Any]]:
        """
        Process single time step of the Extended Object Tracking algorithm.
        
        Main algorithm from paper:
        1. Prediction step for legacy objects
        2. Initialize new potential objects  
        3. Belief propagation (sum-product algorithm)
        4. Pruning and detection
        
        Args:
            measurements: Current measurements (2, M)
            time_step: Current time step
            
        Returns:
            List of detected objects
        """
        # Step 1: Prediction for legacy objects
        self.predict_legacy_objects(self.scan_time)
        
        # Step 2: Initialize new potential objects
        new_objects = self.initialize_new_objects(measurements)
        
        # Step 3: Combine legacy and new objects
        all_objects = self.legacy_objects + new_objects
        
        # Step 4: Belief propagation (sum-product algorithm)
        updated_objects = self.run_belief_propagation(measurements, all_objects)
        
        # Step 5: Pruning
        pruned_objects = self.prune_objects(updated_objects)
        
        # Step 6: Detection
        detections = self.detect_objects(pruned_objects)
        
        # Step 7: Update legacy objects for next time step
        self.legacy_objects = pruned_objects
        
        return detections