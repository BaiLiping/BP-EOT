import numpy as np
from scipy.linalg import cholesky
from copy import deepcopy
from typing import Optional, Any
from scipy.stats import multivariate_normal

import numpy as np
from copy import deepcopy
from typing import Optional, Any

class BernoulliMixture:
    """
    Holds the states,extent,existence probabilities, and labels for a multi-Bernoulli mixture.
    This class is generic and does not require changes for 3D conversion.
    """
    def __init__(self,
                 states: Optional[np.ndarray] = None,
                 extent: Optional[np.ndarray] = None,
                 existence: Optional[Any] = None,
                 label: Optional[Any] = None) -> None:
        self.states = states
        self.extent = extent
        self.existence = list(existence) if existence is not None else []
        self.label = list(label) if label is not None else []

    def copy_from(self, other: "BernoulliMixture") -> None:
        """Deep copy from another BernoulliMixture."""
        self.states = deepcopy(other.states)
        self.extent = deepcopy(other.extent)
        self.existence = deepcopy(other.existence)
        self.label = deepcopy(other.label)

    def size(self) -> int:
        """
        Returns total number of Bernoulli components (assumed to be along the 3rd axis of states).
        """
        if self.states is None or self.states.size == 0:
            return 0
        return self.states.shape[2]

class bp_eo_filter_3d:
    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters
        # Extract parameters with consistent names
        self.mu_n = parameters['mu_n']
        self.mu_c = parameters['mu_c']
        self.f_c = parameters['f_c']
        self.d_t = parameters['d_t']
        self.sensingRange = parameters['measurement_range']
        self.num_particles = parameters['num_particles']
        self.process_noise = parameters['sigma_v']
        self.p_s = parameters['p_s']
        self.num_sensors = parameters['num_sensors']
        self.detection_threshold = parameters['detection_threshold']
        self.pruning_threshold = parameters['pruning_threshold']
        self.num_steps = parameters['nun_steps']
        self.p_d = parameters['p_d']
        self.mu_m = parameters['mu_m']
        self.clutter_intensity = self.mu_c * self.f_c
        self.surveillanceVolume = (4/3) * np.pi * self.sensingRange ** 3
        self.birth_intensity = self.mu_n / self.surveillanceVolume
        self.sensor_positions = parameters['sensor_positions']
        self.var_range = parameters['range_variance']
        self.var_bearing = parameters['bearing_variance']
        self.var_elevation = parameters['elevation_variance']
        self.new_particles = self.initiate_particles([0, 0, 0])
        self.likelihood_table = np.zeros((1, 0, self.num_particles))

        # --- 3D Change: State transition matrix (F) for a 6D state vector ---
        # State vector: [x, y, z, vx, vy, vz]^T
        self.F = np.array([[1, 0, 0, self.d_t, 0, 0],
                           [0, 1, 0, 0, self.d_t, 0],
                           [0, 0, 1, 0, 0, self.d_t],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])

        dt = self.d_t
        self.Q = np.array([[dt**4/4, 0, 0, dt**3/2, 0, 0],
                           [0, dt**4/4, 0, 0, dt**3/2, 0],
                           [0, 0, dt**4/4, 0, 0, dt**3/2],
                           [dt**3/2, 0, 0, dt**2, 0, 0],
                           [0, dt**3/2, 0, 0, dt**2, 0],
                           [0, 0, dt**3/2, 0, 0, dt**2]])

        self.alpha = BernoulliMixture(
            states=np.zeros((6, self.num_particles, 0)),
            extent=np.zeros((3, self.num_particles, 0)),
            existence=[],
            label=[]
        )
        self.varsigma = BernoulliMixture(
            states=np.zeros((6, self.num_particles, 0)),
            extent=np.zeros((3, self.num_particles, 0)),
            existence=[],
            label=[]
        )
        self.gamma = BernoulliMixture(
            states=np.zeros((6, self.num_particles, 0)),
            extent=np.zeros((3, self.num_particles, 0)),
            existence=[],
            label=[]
        )

        self.xi = np.array([])
        self.beta = np.array([])
        self.nu = np.array([])
        self.phi = np.array([])
        self.kappa = np.array([])
        self.iota = np.array([])

    def initiate_particles(self, center: list) -> np.ndarray:
        """
        Placeholder method to generate particles.
        This must be implemented to generate particles with 6D states.
        """
        # Example implementation:
        # initial_state = np.array(center + [0, 0, 0])[:, np.newaxis]
        # return initial_state + np.random.randn(6, self.num_particles)
        return np.zeros((6, self.num_particles))
    

    def compute_alpha(self) -> None:
        """
        Predicts the target states (alpha) from the gamma mixture.
        """
        num_targets = self.gamma.states.shape[2]
        self.alpha.copy_from(self.gamma)

        for target in range(num_targets):
            # Process noise: scale noise by the square root of variance.
            self.alpha.states[:, :, target] = multivariate_normal(self.F @ self.alpha.states[:, :, target], self.Q)
            self.alpha.existence[target] = self.p_s * deepcopy(self.alpha.existence[target])
        self.gamma.copy_from(self.alpha)

    def compute_xi_sigma(self,
                         sensor_measurements: np.ndarray,
                         sensor_index: int) -> None:
        """
        Computes the xi message and updates the varsigma mixture using new sensor measurements.
        """
        num_measurements = sensor_measurements.shape[1] if sensor_measurements.size else 0
        # Access sensor position as a 2D column vector.
        sensor_position = self.sensor_positions[:, sensor_index]
        detection_probability = self.p_d
        clutter_intensity = self.mu_c * self.f_c
        # Use measurement_range parameter if available; otherwise, use sensingRange.
        measurement_range = self.parameters.get('measurement_range', self.sensingRange)
        # Birth intensity computed based on measurement_range.
        birth_intensity = self.mu_n / (2 * np.pi * measurement_range ** 2)

        # Initialize new particles from the given sensor position.
        self.new_particles = self.initiate_particles(sensor_position)

        if num_measurements > 0:
            measurements_likelihood = self.calculate_likelihood_for_new_measurements(sensor_position,
                                                                                    sensor_measurements)
        else:
            measurements_likelihood = np.array([])

        # Allocate space for varsigma mixture and xi message.
        self.varsigma.states = np.empty((4, self.num_particles, num_measurements))
        self.varsigma.label = []
        self.varsigma.existence = []
        self.xi = np.empty((num_measurements,))

        for meas in range(num_measurements):
            self.varsigma.states[:, :, meas] = self.sample_from_likelihood(sensor_measurements[:, meas],
                                                                            sensor_position)
            current_max_label = max(self.gamma.label) if self.gamma.label else 0
            self.varsigma.label.append(current_max_label + meas + 1)
            self.varsigma.existence.append(measurements_likelihood[meas] *
                                             (self.mu_n * detection_probability) / clutter_intensity)
            self.xi[meas] = 1 + deepcopy(self.varsigma.existence[-1])

    def compute_beta(self,
                     sensor_measurements: np.ndarray,
                     sensor_index: int) -> None:
        """
        Evaluates measurement likelihood factors and computes the beta messages.
        """
        var_range = self.var_range
        var_bearing = self.var_bearing
        clutter_intensity = self.f_c * self.mu_c
        detection_probability = self.p_d
        sensor_positions = self.sensor_positions  # 2 x num_sensors
        num_particles = self.num_particles

        num_measurements = sensor_measurements.shape[1] if sensor_measurements.size else 0
        num_targets = self.gamma.states.shape[2] if self.gamma.states.size else 0

        # Initialize beta and likelihood table.
        self.beta = np.zeros((num_measurements + 1, num_targets))
        self.likelihood_table = np.zeros((num_measurements + 1, num_targets, num_particles))

        if num_targets > 0:
            # For missed detections.
            self.likelihood_table[0, :, :] = 1 - detection_probability

            for target in range(num_targets):
                # Compute predicted range and bearing for each particle.
                diff_x = self.gamma.states[0, :, target] - sensor_positions[0, sensor_index]
                diff_y = self.gamma.states[1, :, target] - sensor_positions[1, sensor_index]
                predicted_range = np.sqrt(diff_x ** 2 + diff_y ** 2)
                predicted_bearing = np.degrees(np.arctan2(diff_y, diff_x))

                for m in range(num_measurements):
                    bearing_error = self.wrap_to_180(sensor_measurements[1, m] - predicted_bearing)
                    # Compute likelihood for each particle.
                    self.likelihood_table[m + 1, target, :] = (
                        (1 / (2 * np.pi * np.sqrt(var_bearing * var_range))) *
                        (detection_probability / clutter_intensity) *
                        np.exp(-0.5 * ((sensor_measurements[0, m] - predicted_range) ** 2 / var_range)) *
                        np.exp(-0.5 * (bearing_error ** 2 / var_bearing))
                    )

            # v-factors for missed detections.
            v_factors = np.zeros((num_measurements + 1, num_targets))
            v_factors[0, :] = 1
            # Convert gamma.existence to numpy array for broadcasting.
            existence = np.tile(np.array(self.gamma.existence), (num_measurements + 1, 1))
            self.beta = existence * np.mean(self.likelihood_table, axis=2) + (1 - existence) * v_factors

    def compute_kappa_iota(self):
        """
        Performs iterative belief propagation for data association.
        """
        num_measurements = self.beta.shape[0] - 1
        num_targets = self.beta.shape[1]
        if num_targets == 0 or num_measurements == 0:
            self.kappa = self.beta
            self.iota = self.xi
            return

        # Initialize kappa as ones.
        self.kappa = np.ones((num_measurements, num_targets))
        for iteration in range(100):
            previous_kappa = deepcopy(self.kappa)
            product = self.kappa * self.beta[1:, :]
            likelihood_sum = self.beta[0, :] + np.sum(product, axis=0)
            denominator = np.tile(likelihood_sum, (num_measurements, 1)) - product
            messages = self.beta[1:, :] / (denominator + 1e-12)
            sum_messages = self.xi + np.sum(messages, axis=1)
            self.kappa = 1 / (np.tile(sum_messages[:, None], (1, num_targets)) - messages + 1e-12)
            # Check for convergence in log space.
            distance = np.max(np.abs(np.log(self.kappa + 1e-12) - np.log(previous_kappa + 1e-12)))
            if distance < 1e-5:
                break

        # Combine messages with a column of ones.
        self.iota = np.hstack((np.ones((num_measurements, 1)), messages))
        row_sums = np.sum(self.iota, axis=1, keepdims=True)
        self.iota = self.iota / (row_sums + 1e-12)
        # Extract the first column as the final iota values.
        self.iota = self.iota[:, 0]

    def compute_gamma(self):
        """
        Updates legacy potential tracks and merges them with new tracks.
        """
        num_measurements = self.likelihood_table.shape[0] - 1
        num_particles = self.likelihood_table.shape[2]
        num_targets = self.likelihood_table.shape[1]
  
        for target in range(num_targets):
            # missed detection
            weights = deepcopy(self.likelihood_table[0, target, :])
            # Sum over all measurements.
            for m in range(num_measurements):
                weights += self.kappa[m, target] * self.likelihood_table[m+1, target, :]
            # Sum of the weights of all particles.
            sum_weights = np.sum(weights)
            isAlive = self.gamma.existence[target] * sum_weights / num_particles
            isDead = 1 - self.gamma.existence[target]
            self.gamma.existence[target] = isAlive / (isAlive + isDead + 1e-12)
            if sum_weights > 0:
                norm_weights = weights / sum_weights
                indices = TrackerBP.resample(norm_weights, num_particles)
                self.gamma.states[:, :, target] = deepcopy(self.gamma.states[:, indices, target])
    
        # Merge legacy tracks with new tracks.
        if self.gamma.states.size == 0:
            merged_states = self.varsigma.states
        else:
            merged_states = np.concatenate((self.gamma.states, self.varsigma.states), axis=2)
        self.gamma.states = merged_states
    
        # Update existence for new tracks.
        new_existences = (self.iota * np.array(self.varsigma.existence)) / (self.iota * np.array(self.varsigma.existence) + 1 + 1e-12)
        if not self.gamma.existence:
            merged_existences = new_existences.tolist()
        else:
            merged_existences = self.gamma.existence + new_existences.tolist()
        self.gamma.existence = merged_existences
    
        # Merge labels.
        if not self.gamma.label:
            merged_labels = self.varsigma.label
        else:
            merged_labels = self.gamma.label + self.varsigma.label
        self.gamma.label = merged_labels

    def prune(self):
        """
        Prunes tracks with low existence probability.
        """
        if self.gamma.size() > 0:
            valid_idx = np.array([e >= self.pruning_threshold for e in self.gamma.existence])
            if valid_idx.any():
                self.gamma.states = self.gamma.states[:, :, valid_idx]
                self.gamma.label = np.array(self.gamma.label)[valid_idx].tolist()
                self.gamma.existence = np.array(self.gamma.existence)[valid_idx].tolist()
            else:
                self.gamma.states = np.empty((4, self.num_particles, 0))
                self.gamma.label = []
                self.gamma.existence = []

    def estimate_state(self):
        """
        Estimates the state for targets with existence probability above the detection threshold.
        """
        estimates = {}
        detected_states = []
        detected_labels = []
        detected_existence = []

        if self.gamma.states.shape[2] > 0:
            numTargets = self.gamma.states.shape[2]
            for target in range(numTargets):
                if self.gamma.existence[target] > self.detection_threshold:
                    state_mean = np.mean(self.gamma.states[:, :, target], axis=1)
                    detected_states.append(state_mean)
                    detected_labels.append(self.gamma.label[target])
                    detected_existence.append(self.gamma.existence[target])
        if detected_states:
            estimates['state'] = np.column_stack(detected_states)
            estimates['label'] = np.array(detected_labels)
            estimates['existence'] = np.array(detected_existence)
        estimates['gamma'] = deepcopy(self.gamma)
        return estimates

    def estimate_cardinality(self):
        """
        Estimates the number of targets (cardinality) as the sum of existence probabilities.
        """
        if self.gamma.existence:
            estimated_cardinality = np.sum(self.gamma.existence)
        else:
            estimated_cardinality = 0
        return estimated_cardinality

    def calculate_likelihood_for_new_measurements(self, sensor_position: np.ndarray, sensor_measurements: np.ndarray) -> np.ndarray:
        # Ensure inputs are numpy arrays of type float
        sensor_measurements = np.asarray(sensor_measurements, dtype=float)
        sensor_position = np.asarray(sensor_position, dtype=float)
        self.new_particles = np.asarray(self.new_particles, dtype=float)
        
        measurement_range = self.parameters['measurement_range']
        var_range = float(self.parameters['range_variance'])
        var_bearing = float(self.parameters['bearing_variance'])
                
        # Compute predicted sensor readings for each particle.
        dx = self.new_particles[0, :] - sensor_position[0]
        dy = self.new_particles[1, :] - sensor_position[1]
        predicted_range = np.sqrt(dx**2 + dy**2)
        predicted_bearing = np.degrees(np.arctan2(dy, dx))
        
        # Reshape for broadcasting.
        predicted_range = predicted_range[:, np.newaxis]
        predicted_bearing = predicted_bearing[:, np.newaxis]
        
        # Compute differences.
        range_diff = sensor_measurements[0, np.newaxis, :] - predicted_range
        bearing_diff = sensor_measurements[1, np.newaxis, :] - predicted_bearing
        
        # Compute likelihood (vectorized operations).
        likelihood_particles = (1 / (2 * np.pi * np.sqrt(var_range * var_bearing))) * \
            np.exp(-0.5 * (range_diff**2 / var_range)) * \
            np.exp(-0.5 * (bearing_diff**2 / var_bearing))
        
        likelihood_measurements = np.mean(likelihood_particles, axis=0)
        return likelihood_measurements

    def sample_from_likelihood(self, sensor_measurement: np.ndarray, sensor_position: np.ndarray) -> np.ndarray:
        """
        Samples new potential target states from the measurement likelihood.
        """
        var_range = self.parameters['range_variance']
        var_bearing = self.parameters['bearing_variance']
        priorVelocityCov = self.parameters['velocity_noise']
        num_particles = self.parameters['num_particles']
    
        samples = np.zeros((4, num_particles))
        randomRange = sensor_measurement[0] + np.sqrt(var_range) * np.random.randn(num_particles)
        randomBearing = sensor_measurement[1] + np.sqrt(var_bearing) * np.random.randn(num_particles)
        samples[0, :] = sensor_position[0] + randomRange * np.cos(np.deg2rad(randomBearing))
        samples[1, :] = sensor_position[1] + randomRange * np.sin(np.deg2rad(randomBearing))
        L = cholesky(priorVelocityCov, lower=True)
        samples[2:4, :] = L @ np.random.randn(2, num_particles)
        return samples

    @staticmethod
    def resample(weights: np.ndarray, num_particles: int) -> np.ndarray:
        """
        Systematic resampling algorithm.
        """
        cumWeights = np.cumsum(weights)
        positions = (np.arange(num_particles) + np.random.uniform(0, 1)) / num_particles
        indexes = np.zeros(num_particles, dtype=int)
        i, j = 0, 0
        while i < num_particles:
            if positions[i] < cumWeights[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    @staticmethod
    def wrap_to_180(angle: Any) -> Any:
        """
        Wraps an angle (in degrees) to the interval [-180, 180].
        """
        return ((angle + 180) % 360) - 180

    def initiate_particles(self, sensor_position: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Initializes particles around a sensor position.
        """
        particles = np.zeros((2, self.parameters['num_particles']))
        theta = 360 * np.random.rand(self.parameters['num_particles'])
        r = self.parameters['measurement_range'] * np.sqrt(np.random.rand(self.parameters['num_particles']))
        particles[0, :] = sensor_position[0] + r * np.cos(np.deg2rad(theta))
        particles[1, :] = sensor_position[1] + r * np.sin(np.deg2rad(theta))
        return particles