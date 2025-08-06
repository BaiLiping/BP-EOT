"""
Extended Object Filter implementation based on Meyer & Williams (2021).
"Scalable Detection and Tracking of Geometric Extended Objects"

This class represents a single Extended Object (Potential Object) with:
- 3D kinematic state (position + velocity) 
- 3x3 extent matrix (positive semidefinite)
- Existence probability
- Unique label for tracking
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from scipy.stats import multivariate_normal, wishart, invwishart
from scipy.linalg import inv, sqrtm


class EOFilter:
    """
    Extended Object Filter representing a single Potential Object (PO).
    
    From the paper:
    - Kinematic state x_k = [p_k^T, m_k^T]^T where p_k is position, m_k is velocity
    - Extent state E_k is a symmetric positive semidefinite matrix
    - Existence variable r_k ∈ {0,1} (represented as probability)
    """
    
    def __init__(self, 
                 num_particles: int = 1000,
                 label: Optional[str] = None):
        """
        Initialize an Extended Object Filter.
        
        Args:
            num_particles: Number of particles for particle filter representation
            label: Unique identifier for this object
        """
        # Dimensions for 3D tracking with yaw-only rotation
        self.spatial_dim = 3  # 3D space (x, y, z)
        self.kinematic_dim = 8  # [px, py, pz, vx, vy, vz, yaw_rate, meas_rate]
        self.extent_dim = 3  # 3x3 extent matrix
        
        # Number of particles
        self.num_particles = num_particles
        
        # Particle representation of kinematic state
        # Shape: (8, num_particles) for [px, py, pz, vx, vy, vz, yaw_rate, meas_rate]
        self.kinematic_particles = np.zeros((self.kinematic_dim, num_particles))
        
        # Particle representation of extent state
        # Shape: (3, 3, num_particles) - one 3x3 matrix per particle
        self.extent_particles = np.zeros((self.extent_dim, self.extent_dim, num_particles))
        
        # Initialize extent particles to small positive definite matrices
        for i in range(num_particles):
            # Default to identity-like matrix
            self.extent_particles[:, :, i] = np.eye(self.extent_dim) * 1.0
        
        # Particle weights
        # From paper: weights don't sum to 1, but to existence probability
        self.weights = np.ones(num_particles) / num_particles
        
        # Existence probability p(r_k = 1)
        self.existence_prob = 0.0
        
        # Unique label for tracking
        self.label = label or f"PO_{id(self)}"
        
        # Store alpha messages for BP iterations (Section III-B-2-e)
        self.alpha_messages = None
        
        # Track time of creation
        self.time_created = None
        
        # Track if this is a legacy or new object
        self.is_legacy = False
        
    def set_particles(self, 
                     kinematic_particles: np.ndarray,
                     extent_particles: np.ndarray,
                     weights: Optional[np.ndarray] = None):
        """
        Set particle states directly.
        
        Args:
            kinematic_particles: Array of shape (6, num_particles)
            extent_particles: Array of shape (3, 3, num_particles)
            weights: Optional array of particle weights
        """
        assert kinematic_particles.shape == (self.kinematic_dim, self.num_particles)
        assert extent_particles.shape == (self.extent_dim, self.extent_dim, self.num_particles)
        
        self.kinematic_particles = kinematic_particles.copy()
        self.extent_particles = extent_particles.copy()
        
        if weights is not None:
            assert weights.shape == (self.num_particles,)
            self.weights = weights.copy()
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles
    
    def get_mmse_estimate(self) -> Dict[str, np.ndarray]:
        """
        Compute MMSE estimates of state.
        From paper equation (2): MMSE estimate is weighted average.
        
        Returns:
            Dictionary with 'position', 'velocity', 'extent' estimates
        """
        # Normalize weights for MMSE computation
        normalized_weights = self.weights / np.sum(self.weights)
        
        # MMSE kinematic state
        kinematic_mmse = np.average(self.kinematic_particles, 
                                   axis=1, 
                                   weights=normalized_weights)
        
        # MMSE extent state
        extent_mmse = np.average(self.extent_particles, 
                                axis=2, 
                                weights=normalized_weights)
        
        return {
            'position': kinematic_mmse[:3],          # [px, py, pz]
            'velocity': kinematic_mmse[3:6],         # [vx, vy, vz]
            'yaw_rate': kinematic_mmse[6],           # yaw rate (rotation around z-axis)
            'meas_rate': kinematic_mmse[7],          # measurement rate
            'extent': extent_mmse,
            'existence_prob': self.existence_prob,
            'label': self.label
        }
    
    def copy(self) -> 'EOFilter':
        """
        Create a deep copy of this filter.
        
        Returns:
            New EOFilter instance with copied state
        """
        new_filter = EOFilter(num_particles=self.num_particles, 
                             label=f"{self.label}_copy")
        
        new_filter.kinematic_particles = self.kinematic_particles.copy()
        new_filter.extent_particles = self.extent_particles.copy()
        new_filter.weights = self.weights.copy()
        new_filter.existence_prob = self.existence_prob
        new_filter.time_created = self.time_created
        new_filter.is_legacy = self.is_legacy
        
        if self.alpha_messages is not None:
            new_filter.alpha_messages = self.alpha_messages.copy()
        
        return new_filter
    
    def __repr__(self) -> str:
        """String representation of the filter."""
        mmse = self.get_mmse_estimate()
        pos = mmse['position']
        return (f"EOFilter(label={self.label}, "
                f"pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}], "
                f"exist_prob={self.existence_prob:.3f})")
    
    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================
    
    @staticmethod
    def sample_inverse_wishart(dof: float, scale_matrix: np.ndarray) -> np.ndarray:
        """
        Sample from Inverse Wishart distribution IW(ν, Ψ).
        
        The Inverse Wishart is the conjugate prior for covariance matrices.
        If E ~ IW(ν, Ψ), then:
        - E is positive definite
        - E[E] = Ψ / (ν - d - 1) for ν > d + 1
        
        Uses scipy.stats.invwishart for proper sampling.
        
        Args:
            dof: Degrees of freedom ν (must be > dimension - 1)
            scale_matrix: Scale matrix Ψ (positive definite)
            
        Returns:
            Sample from IW(ν, Ψ)
        """
        d = scale_matrix.shape[0]
        
        # Validate inputs
        if dof <= d - 1:
            raise ValueError(f"Degrees of freedom {dof} must be > dimension - 1 = {d-1}")
        
        try:
            # Ensure scale matrix is positive definite
            eigenvals = np.linalg.eigvalsh(scale_matrix)
            if np.min(eigenvals) <= 0:
                # Make it positive definite
                scale_matrix = scale_matrix + np.eye(d) * (abs(np.min(eigenvals)) + 1e-6)
            
            # Use scipy's native inverse Wishart implementation
            # Note: scipy.stats.invwishart uses (df, scale) parameterization
            return invwishart.rvs(df=dof, scale=scale_matrix)
            
        except (np.linalg.LinAlgError, ValueError):
            # Fallback: use the Wishart inversion method if native fails
            try:
                scale_inv = inv(scale_matrix)
                wishart_sample = wishart.rvs(df=dof, scale=scale_inv)
                return inv(wishart_sample)
            except:
                # Last resort: return scaled matrix
                return scale_matrix / dof
    
    # ========================================================================
    # MESSAGE COMPUTATION FUNCTIONS - Following Meyer & Williams (2021)
    # ========================================================================
    
    def compute_alpha(self, 
                     dt: float,
                     survival_prob: float,
                     measurement_rate_func: callable,
                     motion_params: Optional[Dict] = None) -> Tuple[np.ndarray, float]:
        """
        Compute α (alpha) message - Prediction step with 3D CTRV motion model.
        
        Paper reference: Section III-B-1, Equations (10)-(12) and Equation (8)
        
        Implements the state transition from Equation (8):
        f(x_k, E_k | x_{k-1}, E_{k-1}) = N(x_k; f(x_{k-1}), Σ_k) × W(E_k; q_k, V(m_{k-1}) E_{k-1} V(m_{k-1})^T / q_k)
        
        Where:
        - f(x_{k-1}) is the 3D CTRV (Constant Turn Rate and Velocity) model
        - V(m_{k-1}) is yaw rotation matrix (rotation around z-axis only)
        - Extent is rotated: V * E * V^T
        
        Kinematic state: [px, py, pz, vx, vy, vz, yaw_rate, meas_rate]
        
        Args:
            dt: Time step duration
            survival_prob: p_s - probability of object survival
            measurement_rate_func: Function to compute μ_m(x,e)
            motion_params: Dict with motion model parameters
            
        Returns:
            Tuple of (predicted_weights, alpha_n) where alpha_n is for r_k=0
        """
        # Default motion parameters
        if motion_params is None:
            motion_params = {
                'process_noise_pos': 1.0,      # Process noise for position (x,y,z)
                'process_noise_vel': 1.0,      # Process noise for velocity (vx,vy,vz)
                'process_noise_yaw': 0.1,      # Process noise for yaw rate
                'process_noise_meas': 0.1,     # Process noise for measurement rate
                'extent_dof': 10.0,            # Degrees of freedom for Wishart
            }
        
        # Extract motion parameters
        σ_pos = motion_params.get('process_noise_pos', 1.0)
        σ_vel = motion_params.get('process_noise_vel', 1.0)
        σ_yaw = motion_params.get('process_noise_yaw', 0.1)
        σ_meas = motion_params.get('process_noise_meas', 0.1)
        q_extent = motion_params.get('extent_dof', 10.0)
        
        # Predict particles
        predicted_kinematic = np.zeros_like(self.kinematic_particles)
        predicted_extent = np.zeros_like(self.extent_particles)
        
        for j in range(self.num_particles):
            # Current state: [px, py, pz, vx, vy, vz, yaw_rate, meas_rate]
            prev_state = self.kinematic_particles[:, j]
            px_prev, py_prev, pz_prev, vx_prev, vy_prev, vz_prev, yaw_rate_prev, meas_rate_prev = prev_state
            
            # 3D CTRV Motion Model (Constant Turn Rate and Velocity)
            # Yaw affects only x-y plane motion, z motion is independent
            
            if abs(yaw_rate_prev) < 1e-6:  # Nearly straight motion
                # Linear motion in x-y plane
                px_pred = px_prev + vx_prev * dt
                py_pred = py_prev + vy_prev * dt
                vx_pred = vx_prev
                vy_pred = vy_prev
            else:
                # Curved motion in x-y plane with yaw rate
                # CTRV equations from standard literature (x-y plane only)
                sin_wt = np.sin(yaw_rate_prev * dt)
                cos_wt = np.cos(yaw_rate_prev * dt)
                
                px_pred = px_prev + (vx_prev * sin_wt - vy_prev * (cos_wt - 1)) / yaw_rate_prev
                py_pred = py_prev + (vy_prev * sin_wt + vx_prev * (cos_wt - 1)) / yaw_rate_prev
                vx_pred = vx_prev * cos_wt - vy_prev * sin_wt
                vy_pred = vy_prev * cos_wt + vx_prev * sin_wt
            
            # Z motion is independent (constant velocity)
            pz_pred = pz_prev + vz_prev * dt
            vz_pred = vz_prev
            
            # Yaw rate and measurement rate evolution with noise
            yaw_rate_pred = yaw_rate_prev + np.random.normal(0, σ_yaw * dt)
            meas_rate_pred = max(0.1, meas_rate_prev + np.random.normal(0, σ_meas * dt))
            
            # Add process noise to position and velocity
            px_pred += np.random.normal(0, σ_pos * dt)
            py_pred += np.random.normal(0, σ_pos * dt)
            pz_pred += np.random.normal(0, σ_pos * dt)
            vx_pred += np.random.normal(0, σ_vel * dt)
            vy_pred += np.random.normal(0, σ_vel * dt)
            vz_pred += np.random.normal(0, σ_vel * dt)
            
            predicted_kinematic[:, j] = [px_pred, py_pred, pz_pred, vx_pred, vy_pred, vz_pred, yaw_rate_pred, meas_rate_pred]
            
            # Extent Prediction with Yaw Rotation (Equation 8)
            # E_k ~ W(q_k, V(m_{k-1}) E_{k-1} V(m_{k-1})^T / q_k)
            prev_extent = self.extent_particles[:, :, j]
            
            # Yaw rotation matrix V(yaw_rate) - rotation around z-axis only
            yaw_angle = yaw_rate_prev * dt
            cos_yaw = np.cos(yaw_angle)
            sin_yaw = np.sin(yaw_angle)
            
            # 3x3 rotation matrix for yaw (rotation around z-axis)
            V_yaw = np.array([[cos_yaw, -sin_yaw, 0],
                             [sin_yaw,  cos_yaw, 0],
                             [0,        0,       1]])
            
            # Rotate extent: V * E * V^T (yaw rotation only)
            rotated_extent = V_yaw @ prev_extent @ V_yaw.T
            
            # Sample from Wishart distribution for extent prediction
            # W(q, Ψ) where Ψ = rotated_extent / q
            scale_matrix = rotated_extent / q_extent
            
            try:
                # Sample from Wishart distribution
                # Note: Using Wishart (not Inverse Wishart) as per Equation (8)
                predicted_extent[:, :, j] = wishart.rvs(df=q_extent, scale=scale_matrix)
            except (ValueError, np.linalg.LinAlgError):
                # Fallback if sampling fails
                predicted_extent[:, :, j] = rotated_extent
            
            # Ensure positive semidefinite
            eigenvals, eigenvecs = np.linalg.eigh(predicted_extent[:, :, j])
            eigenvals = np.maximum(eigenvals, 1e-6)
            predicted_extent[:, :, j] = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Update particles
        self.kinematic_particles = predicted_kinematic
        self.extent_particles = predicted_extent
        
        # Compute weights according to equation (11)
        # w̲_k^(j) = p_s * exp(-μ_m(x̲_k^(j), ē_k^(j))) * w^-_k^(j)
        predicted_weights = np.zeros(self.num_particles)
        for j in range(self.num_particles):
            mu_m = measurement_rate_func(self.kinematic_particles[:, j], 
                                        self.extent_particles[:, :, j])
            predicted_weights[j] = survival_prob * np.exp(-mu_m) * self.weights[j]
        
        # Compute α̲^n_k according to equation (12)
        p_exist_prev = np.sum(self.weights)
        alpha_n = (1 - p_exist_prev) + (1 - survival_prob) * p_exist_prev
        
        # Update weights
        self.weights = predicted_weights
        
        return predicted_weights, alpha_n
    
    def compute_beta(self,
                    measurement: np.ndarray,
                    measurement_likelihood_func: callable,
                    measurement_rate_func: callable,
                    mu_fa: float,
                    f_fa: float,
                    alpha_kl: Optional[np.ndarray] = None) -> float:
        """
        Compute β (beta) message - Measurement evaluation.
        
        Paper reference: Section III-B-2-a, Equation (15)
        β_{kl}^(p)(b_l = k) = (1 / (μ_fa * f_fa(z_l) * α_{kl}^(p))) * 
                              ∫∫ μ_m(x̲_k, ē_k) * f(z_l | x̲_k, ē_k) * α_{kl}^(p)(x̲_k, ē_k, r̲_k=1) dx̲_k dē_k
        
        Args:
            measurement: z_l - measurement vector (3D position)
            measurement_likelihood_func: Function to compute f(z_l | x_k, e_k)
            measurement_rate_func: Function to compute μ_m(x_k, e_k)
            mu_fa: μ_fa - false alarm rate
            f_fa: f_fa - false alarm PDF value
            alpha_kl: α_{kl}^(p) weights (if None, use current weights)
            
        Returns:
            β_{kl} message value
        """
        if alpha_kl is None:
            alpha_kl = self.weights
            
        # Compute integral using Monte Carlo integration over particles
        # Section V-B: Particle-based approximation
        integral_sum = 0.0
        alpha_sum = np.sum(alpha_kl)
        
        for j in range(self.num_particles):
            # Extract particle state
            x_j = self.kinematic_particles[:, j]
            e_j = self.extent_particles[:, :, j]
            w_j = alpha_kl[j]
            
            # Compute measurement rate μ_m(x_k^(j), e_k^(j))
            mu_m = measurement_rate_func(x_j, e_j)
            
            # Compute likelihood f(z_l | x_k^(j), e_k^(j))
            likelihood = measurement_likelihood_func(measurement, x_j, e_j)
            
            # Add weighted contribution
            integral_sum += w_j * mu_m * likelihood
        
        # Compute β message according to equation (15)
        if alpha_sum > 0:
            beta_kl = integral_sum / (mu_fa * f_fa * alpha_sum)
        else:
            beta_kl = 1e-10  # Small value to avoid numerical issues
            
        return beta_kl
    
    def compute_nu(self,
                  beta_messages_from_others: List[float],
                  beta_messages_from_new: List[float]) -> float:
        """
        Compute ν (nu) message - Data association.
        
        Paper reference: Section III-B-2-b, Equation (18)
        ν_{lk}^(p)(b_l) = ∏_{k'≠k} β_{k'l}^(p)(b_l) * ∏_{ℓ=1}^l β̄_{ℓl}^(p)(b_l)
        
        This is the product of all β messages except the one from current object k.
        
        Args:
            beta_messages_from_others: List of β messages from other legacy objects
            beta_messages_from_new: List of β messages from new objects
            
        Returns:
            ν message value
        """
        # Product of all β messages from other objects
        nu_value = 1.0
        
        # Multiply β messages from other legacy objects
        for beta in beta_messages_from_others:
            nu_value *= beta
            
        # Multiply β messages from new objects
        for beta in beta_messages_from_new:
            nu_value *= beta
            
        return nu_value
    
    def compute_gamma(self,
                     measurement: np.ndarray,
                     measurement_likelihood_func: callable,
                     measurement_rate_func: callable,
                     mu_fa: float,
                     f_fa: float,
                     xi_kl: float) -> Tuple[np.ndarray, float]:
        """
        Compute γ (gamma) message - Measurement update.
        
        Paper reference: Section III-B-2-c, Equation (20)
        γ_{lk}^(p)(x̲_k, ē_k, r̲_k=1) = [μ_m(x̲_k, ē_k) * f(z_l | x̲_k, ē_k) / (μ_fa * f_fa(z_l))] + ξ_{kl}^(p)
        γ_{lk}^(p)(x̲_k, ē_k, r̲_k=0) = ξ_{kl}^(p)
        
        Args:
            measurement: z_l - measurement vector
            measurement_likelihood_func: Function to compute f(z_l | x_k, e_k)
            measurement_rate_func: Function to compute μ_m(x_k, e_k)
            mu_fa: μ_fa - false alarm rate
            f_fa: f_fa - false alarm PDF value
            xi_kl: ξ_{kl}^(p) - sum from equation (21)
            
        Returns:
            Tuple of (gamma_weights_r1, gamma_r0) for r_k=1 and r_k=0
        """
        gamma_weights = np.zeros(self.num_particles)
        
        # For each particle, compute γ contribution for r_k = 1
        for j in range(self.num_particles):
            x_j = self.kinematic_particles[:, j]
            e_j = self.extent_particles[:, :, j]
            
            # Measurement rate and likelihood
            mu_m = measurement_rate_func(x_j, e_j)
            likelihood = measurement_likelihood_func(measurement, x_j, e_j)
            
            # γ message for this particle according to equation (20)
            gamma_weights[j] = (mu_m * likelihood) / (mu_fa * f_fa) + xi_kl
        
        # γ for r_k = 0 is just ξ
        gamma_r0 = xi_kl
        
        return gamma_weights, gamma_r0
    
    def compute_xi(self,
                  nu_messages: List[float],
                  beta_self: float) -> float:
        """
        Compute ξ (xi) - auxiliary sum for measurement update.
        
        Paper reference: Section III-B-2-c, Equation (21)
        ξ_{kl}^(p) = ∑_{b_l≠k} ν_{lk}^(p)(b_l)
        
        This represents the sum of all ν messages except for b_l = k.
        
        Args:
            nu_messages: List of ν message values for different associations
            beta_self: β message from this object (to exclude)
            
        Returns:
            ξ value
        """
        # Sum all ν messages (which already exclude self)
        # In implementation, this is typically the sum of products of other β messages
        xi_sum = np.sum(nu_messages)
        
        # Ensure non-negative
        return max(xi_sum, 0.0)
    
    def update_belief(self,
                     alpha_weights: np.ndarray,
                     gamma_weights_list: List[np.ndarray],
                     alpha_n: float,
                     xi_product: float) -> None:
        """
        Update belief after message passing.
        
        Paper reference: Section III-B-2-d, Equations (24)-(25)
        For legacy objects - Equation (24):
        f̃(x̲_k, ē_k, 1) ∝ α(x̲_k, ē_k, 1) * ∏_l γ_{lk}^(P)
        f̃_k = α^n_k * ∏_l ξ_{kl}^(P)
        
        Args:
            alpha_weights: α message weights for particles
            gamma_weights_list: List of γ message weights from each measurement
            alpha_n: α^n value for non-existence
            xi_product: Product of all ξ values
        """
        # Update weights for r_k = 1 according to equation (24)
        updated_weights = alpha_weights.copy()
        
        # Multiply by all γ messages
        for gamma_weights in gamma_weights_list:
            updated_weights *= gamma_weights
            
        # Normalize to avoid numerical issues (but maintain existence probability)
        max_weight = np.max(updated_weights)
        if max_weight > 0:
            updated_weights = updated_weights / max_weight
            
        self.weights = updated_weights
        
        # Update existence probability
        # p(r_k = 1) = sum(weights) / (sum(weights) + f̃_k)
        weight_sum = np.sum(self.weights)
        f_tilde_k = alpha_n * xi_product
        
        if weight_sum + f_tilde_k > 0:
            self.existence_prob = weight_sum / (weight_sum + f_tilde_k)
        else:
            self.existence_prob = 0.0
    
    def compute_extrinsic_alpha(self,
                               gamma_weights: np.ndarray,
                               current_alpha: np.ndarray) -> np.ndarray:
        """
        Compute extrinsic information α message for next iteration.
        
        Paper reference: Section III-B-2-e, Equations (26)-(27)
        α_{kl}^(p+1)(x̲_k, ē_k, r̲_k=1) = α(x̲_k, ē_k, 1) * ∏_{l'≠l} γ_{l'k}^(p)
        
        This is the belief divided by the contribution from measurement l.
        
        Args:
            gamma_weights: γ weights from measurement l to exclude
            current_alpha: Current α weights
            
        Returns:
            Extrinsic α weights for next iteration
        """
        # Extrinsic information: divide belief by γ from measurement l
        # In log space for numerical stability
        log_alpha = np.log(current_alpha + 1e-10)
        log_gamma = np.log(gamma_weights + 1e-10)
        
        # Subtract in log space (divide in normal space)
        log_extrinsic = log_alpha - log_gamma
        
        # Convert back to normal space
        extrinsic_alpha = np.exp(log_extrinsic)
        
        # Normalize to avoid numerical issues
        max_weight = np.max(extrinsic_alpha)
        if max_weight > 0:
            extrinsic_alpha = extrinsic_alpha / max_weight
            
        return extrinsic_alpha