import numpy as np
from scipy.stats import chi2

# Placeholder functions for domain-specific computations.
# You will need to provide implementations for these based on your existing MATLAB code.
def get_log_weights_fast(measurement, particles_position, covariance_stack):
    """
    Compute log-weights for each particle given a measurement.
    :param measurement: array of shape (2,)
    :param particles_position: array of shape (2, num_particles)
    :param covariance_stack: array of shape (2, 2, num_particles)
    :return: array of log-weights of length num_particles
    """
    raise NotImplementedError


def get_square2_fast(extent):
    """
    Compute the square of the extent for each particle.
    :param extent: array of shape (2, 2, num_particles)
    :return: array of shape (2, 2, num_particles)
    """
    raise NotImplementedError


def update_particles(particles_kin, particles_ext, existence, log_weights, parameters):
    """
    Update particles' kinematic states, extents, and existence probability.
    :return: updated (particles_kin, particles_ext, existence)
    """
    raise NotImplementedError


def get_promising_new_targets(curr_kin, curr_ext, curr_exist, measurements, params):
    """
    Identify promising measurements as new target initiations.

    :param curr_kin: np.ndarray, shape (state_dim, num_particles, num_targets)
    :param curr_ext: np.ndarray, shape (2, 2, num_particles, num_targets)
    :param curr_exist: np.ndarray, shape (num_targets,)
    :param measurements: np.ndarray, shape (2, num_measurements)
    :param params: dict with keys:
        - measurementVariance: float
        - surveillanceRegion: [[x_min, y_min], [x_max, y_max]]
        - meanMeasurements: float
        - meanClutter: float
        - (plus any keys needed by update_particles)
    :return: (central_indexes, reordered_measurements)
    """
    num_measurements = measurements.shape[1]
    num_particles = curr_kin.shape[1]

    # Measurement noise covariance (2x2)
    meas_cov = params['measurementVariance'] * np.eye(2)

    # Compute surveillance area
    (x_min, y_min), (x_max, y_max) = params['surveillanceRegion']
    area_size = (x_max - x_min) * (y_max - y_min)

    # Clutter-adjusted constant factor
    mean_meas = params['meanMeasurements']
    mean_clutter = params['meanClutter']
    const_factor = area_size * (mean_meas / mean_clutter)

    probabilities_new = np.ones(num_measurements)

    # Loop over each measurement
    for m in range(num_measurements):
        num_targets = curr_kin.shape[2]

        # Prepare data-association inputs
        input_da = np.ones((2, num_targets))
        likelihoods = np.zeros((num_particles, num_targets))

        for t in range(num_targets):
            # Build per-particle covariance including extent
            extent_cov = get_square2_fast(curr_ext[:, :, :, t])
            cov_stack = extent_cov + np.tile(meas_cov[:, :, np.newaxis], (1, 1, num_particles))

            # Compute log-likelihoods for this target
            log_w = get_log_weights_fast(measurements[:, m], curr_kin[0:2, :, t], cov_stack)
            likelihoods[:, t] = const_factor * np.exp(log_w)

            # Expected likelihood (mean over particles)
            input_da[1, t] = np.mean(likelihoods[:, t])

            # If target already exists, weight by its existence probability
            input_da[:, t] = curr_exist[t] * input_da[:, t] + (1 - curr_exist[t]) * np.array([1, 0])

        # Compute probability that this measurement is from a new target
        ratio = input_da[1, :] / input_da[0, :]
        sum_da = 1 + np.sum(ratio)
        output_da = 1.0 / (sum_da - ratio)
        probabilities_new[m] = 1.0 / sum_da

        # On the last measurement, skip updating existing targets
        if m == num_measurements - 1:
            break

        # Update particle sets for each existing target
        for t in range(num_targets):
            w = np.log(1 + likelihoods[:, t] * output_da[t])
            curr_kin[:, :, t], curr_ext[:, :, :, t], curr_exist[t] = update_particles(
                curr_kin[:, :, t],
                curr_ext[:, :, :, t],
                curr_exist[t],
                w,
                params
            )

    # Reorder measurements and select central indexes for clustering
    central_indexes, reorder_idx = get_central_reordered(measurements, probabilities_new, params)
    reordered_measurements = measurements[:, reorder_idx]
    return central_indexes, reordered_measurements


def get_central_reordered(measurements, probs_new, params):
    """
    Cluster 'free' measurements and select central ones.

    :param measurements: np.ndarray, shape (2, N)
    :param probs_new: np.ndarray, shape (N,)
    :param params: dict (will be updated with clustering thresholds)
    :return: (central_indexes, indexes_reordered)
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
    mean_extent_birth = (params['priorExtent1'] / (params['priorExtent2'] - 3)) ** 2
    birth_cov = params['measurementVariance'] * np.eye(2) + mean_extent_birth

    # Perform clustering
    clusters = get_clusters(free_meas, birth_cov, cluster_th)

    # Determine clusters with enough elements
    sizes = [len(c) for c in clusters]
    valid_clusters = [c for c in clusters if len(c) >= min_cluster_elems]
    unused = [idx for idx in free_idx if not any(idx in c for c in valid_clusters)]

    # For each valid cluster, find the central measurement
    central_idxs = []
    ordered_free = []
    for cluster in valid_clusters:
        pts = free_meas[:, cluster]
        if len(cluster) > 1:
            # Compute pairwise Mahalanobis distances
            D = np.zeros((len(cluster), len(cluster)))
            for i in range(len(cluster)):
                for j in range(i+1, len(cluster)):
                    diff = pts[:, i] - pts[:, j]
                    D[i, j] = np.sqrt(diff.T @ np.linalg.inv(birth_cov) @ diff)
                    D[j, i] = D[i, j]
            # Select the point with the largest total distance
            sums = D.sum(axis=1)
            center_idx = cluster[np.argmax(sums)]
        else:
            center_idx = cluster[0]
        central_idxs.append(center_idx)
        # Order remaining cluster points after central
        ordered_free.extend([i for i in cluster if i != center_idx])

    # Build final reorder list: unused free, then cluster centers + others, then assigned
    idx_reordered = unused + central_idxs + ordered_free + assigned_idx.tolist()
    return np.array(central_idxs), np.array(idx_reordered)


def get_clusters(measurements, cov, threshold_prob):
    """
    Agglomerative clustering based on Mahalanobis distance.

    :return: list of clusters, each a list of measurement indices
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

