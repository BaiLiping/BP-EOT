import numpy as np
from scipy.stats import chi2


def get_start_states(num_targets, radius, speed, parameters):
    prior_extent2 = parameters["priorExtent2"]
    prior_extent1 = parameters["priorExtent1"]

    start_matrixes = np.zeros((2, 2, num_targets))
    for target in range(num_targets):
        start_matrixes[:, :, target] = iwishrnd(prior_extent1, prior_extent2)

    if num_targets < 2:
        start_states = np.zeros((4, 1))
        start_states[2] = speed
    else:
        start_states = np.zeros((4, num_targets))
        start_states[:, 0] = [0, radius, 0, -speed]

        step_size = 2 * np.pi / num_targets
        angle = 0
        for target in range(1, num_targets):
            angle += step_size
            start_states[:, target] = [
                np.sin(angle) * radius,
                np.cos(angle) * radius,
                -np.sin(angle) * speed,
                -np.cos(angle) * speed,
            ]

    return start_states, start_matrixes


def generate_tracks_unknown(
    parameters, start_states, extent_matrixes, appearance_from_to, num_steps
):
    acceleration_deviation = parameters["accelerationDeviation"]
    scan_time = parameters["scanTime"]
    num_targets = start_states.shape[1]

    A, W = get_transition_matrices(scan_time)
    target_tracks = np.full((4, num_steps, num_targets), np.nan)
    target_extents = np.full((2, 2, num_steps, num_targets), np.nan)

    for target in range(num_targets):
        tmp = start_states[:, target][:, np.newaxis]
        for step in range(num_steps):
            tmp = A @ tmp + W @ (
                acceleration_deviation * np.random.randn(2, 1)
            )
            if (step >= appearance_from_to[target, 0]) and (
                step <= appearance_from_to[target, 1]
            ):
                target_tracks[:, step, target] = tmp.flatten()
                target_extents[:, :, step, target] = extent_matrixes[:, :, target]

    return target_tracks, target_extents


def generate_cluttered_measurements(target_tracks, target_extents, parameters):
    mean_clutter = parameters["meanClutter"]
    measurement_variance = parameters["measurementVariance"]
    mean_measurements = parameters["meanMeasurements"]
    surveillance_region = parameters["surveillanceRegion"]

    num_steps = target_tracks.shape[1]
    num_targets = target_tracks.shape[2]

    cluttered_measurements = [None] * num_steps
    for step in range(num_steps):
        measurements = np.zeros((2, 0))
        num_measurements = 0
        for target in range(num_targets):
            if np.isnan(target_tracks[0, step, target]):
                continue

            num_measurements_tmp = np.random.poisson(mean_measurements)
            measurements_tmp1 = (
                np.tile(
                    target_tracks[0:2, step, target][:, np.newaxis],
                    (1, num_measurements_tmp),
                )
                + np.random.multivariate_normal(
                    np.zeros(2),
                    target_extents[:, :, step, target] ** 2,
                    num_measurements_tmp,
                ).T
            )
            measurements_tmp1 += np.sqrt(measurement_variance) * np.random.randn(
                2, num_measurements_tmp
            )
            measurements = np.concatenate((measurements, measurements_tmp1), axis=1)
            num_measurements += num_measurements_tmp

        num_false_alarms = np.random.poisson(mean_clutter)
        if num_false_alarms is None:
            num_false_alarms = 0

        false_alarms = np.zeros((2, num_false_alarms))
        false_alarms[0, :] = (
            surveillance_region[1, 0] - surveillance_region[0, 0]
        ) * np.random.rand(1, num_false_alarms) + surveillance_region[0, 0]
        false_alarms[1, :] = (
            surveillance_region[1, 1] - surveillance_region[0, 1]
        ) * np.random.rand(1, num_false_alarms) + surveillance_region[0, 1]

        measurements = np.concatenate((false_alarms, measurements), axis=1)
        measurements = measurements[
            :, np.random.permutation(num_false_alarms + num_measurements)
        ]
        cluttered_measurements[step] = measurements

    return cluttered_measurements


def show_results(
    target_tracks,
    target_extents,
    estimated_tracks,
    estimated_extents,
    measurements_cell,
    axis_values,
    visualization_mode,
):
    # This function requires a GUI library like Matplotlib to be implemented.
    # For now, it's a placeholder.
    pass


def perform_prediction(
    old_particles, old_existences, old_extents, scan_time, parameters
):
    _, num_particles, num_targets = old_particles.shape
    driving_noise_variance = parameters["accelerationDeviation"] ** 2
    survival_probability = parameters["survivalProbability"]
    degree_freedom_prediction = parameters["degreeFreedomPrediction"]

    A, W = get_transition_matrices(scan_time)
    new_particles = old_particles.copy()
    new_existences = old_existences.copy()
    new_extents = old_extents.copy()

    for target in range(num_targets):
        old_extents[:, :, :, target] = (
            old_extents[:, :, :, target] / degree_freedom_prediction
        )
        new_extents[:, :, :, target] = iwishrnd_fast_vector(
            old_extents[:, :, :, target], degree_freedom_prediction, num_particles
        )

    for target in range(num_targets):
        new_particles[:, :, target] = A @ old_particles[:, :, target] + W @ (
            np.sqrt(driving_noise_variance) * np.random.randn(2, num_particles)
        )
        new_existences[target] = survival_probability * old_existences[target]

    return new_particles, new_existences, new_extents


def get_promising_new_targets(
    current_particles_kinematic_tmp,
    current_particles_extent_tmp,
    current_existences_tmp,
    measurements,
    parameters,
):
    num_measurements = measurements.shape[1]
    num_particles = current_particles_kinematic_tmp.shape[1]
    measurements_covariance = parameters["measurementVariance"] * np.eye(2)
    surveillance_region = parameters["surveillanceRegion"]
    area_size = (surveillance_region[1, 0] - surveillance_region[0, 0]) * (
        surveillance_region[1, 1] - surveillance_region[0, 1]
    )
    mean_measurements = parameters["meanMeasurements"]
    mean_clutter = parameters["meanClutter"]
    constant_factor = area_size * (mean_measurements / mean_clutter)

    probabilities_new = np.ones((num_measurements, 1))
    for measurement in range(num_measurements):
        num_targets = current_particles_kinematic_tmp.shape[2]
        input_da = np.ones((2, num_targets))
        likelihoods = np.zeros((num_particles, num_targets))

        for target in range(num_targets):
            likelihoods[:, target] = constant_factor * np.exp(
                get_log_weights_fast(
                    measurements[:, measurement],
                    current_particles_kinematic_tmp[0:2, :, target],
                    get_square2_fast(current_particles_extent_tmp[:, :, :, target])
                + measurements_covariance[:,:,np.newaxis],
                )
            )
            input_da[1, target] = np.mean(likelihoods[:, target], axis=0)
            input_da[:, target] = current_existences_tmp[target] * input_da[
                :, target
            ] + (1 - current_existences_tmp[target]) * np.array([1, 0])

        input_da = input_da[1, :] / input_da[0, :]
        sum_input_da = 1 + np.sum(input_da, axis=0)
        output_da = 1 / (np.tile(sum_input_da, (1, num_targets)) - input_da)
        probabilities_new[measurement] = 1 / sum_input_da

        if measurement == num_measurements - 1:
            break

        for target in range(num_targets):
            log_weights = np.log(
                np.ones((num_particles, 1))
                + likelihoods[:, target] * output_da[0, target]
            )
            (
                current_particles_kinematic_tmp[:, :, target],
                current_particles_extent_tmp[:, :, :, target],
                current_existences_tmp[target],
            ) = update_particles(
                current_particles_kinematic_tmp[:, :, target],
                current_particles_extent_tmp[:, :, :, target],
                current_existences_tmp[target],
                log_weights,
                parameters,
            )

    central_indexes, indexes_reordered = get_central_reordered(
        measurements, probabilities_new, parameters
    )
    measurements = measurements[:, indexes_reordered]
    return central_indexes, measurements


def get_log_weights_fast(
    measurement, current_particles_kinematic, current_particles_extent
):
    num_particles = current_particles_extent.shape[2]
    all_determinantes = (
        current_particles_extent[0, 0, :] * current_particles_extent[1, 1, :]
        - current_particles_extent[0, 1, :] ** 2
    )
    all_factors = np.log(1 / (2 * np.pi * np.sqrt(all_determinantes)))

    measurements_reptition = np.tile(measurement[:,np.newaxis], (1, num_particles))
    part2 = (measurements_reptition - current_particles_kinematic[0:2, :]).T

    tmp = (
        1
        / np.tile(all_determinantes, (2, 1)).T
        * (measurements_reptition.T - current_particles_kinematic[0:2, :].T)
    )
    part1 = np.zeros((num_particles, 2))
    part1[:, 0] = tmp[:, 0] * np.squeeze(current_particles_extent[1, 1, :]) - tmp[
        :, 1
    ] * np.squeeze(current_particles_extent[1, 0, :])
    part1[:, 1] = -tmp[:, 0] * np.squeeze(current_particles_extent[0, 1, :]) + tmp[
        :, 1
    ] * np.squeeze(current_particles_extent[0, 0, :])

    log_weights = all_factors + (
        -1 / 2 * (part1[:, 0] * part2[:, 0] + part1[:, 1] * part2[:, 1])
    )
    return log_weights


def get_square2_fast(matrixes_in):
    matrixes_out = matrixes_in.copy()
    matrixes_out[0, 0, :] = (
        matrixes_in[0, 0, :] * matrixes_in[0, 0, :]
        + matrixes_in[1, 0, :] * matrixes_in[1, 0, :]
    )
    matrixes_out[1, 0, :] = (
        matrixes_in[1, 0, :] * matrixes_in[0, 0, :]
        + matrixes_in[1, 1, :] * matrixes_in[1, 0, :]
    )
    matrixes_out[0, 1, :] = (
        matrixes_in[0, 0, :] * matrixes_in[0, 1, :]
        + matrixes_in[0, 1, :] * matrixes_in[1, 1, :]
    )
    matrixes_out[1, 1, :] = (
        matrixes_in[1, 0, :] * matrixes_in[0, 1, :]
        + matrixes_in[1, 1, :] * matrixes_in[1, 1, :]
    )
    return matrixes_out


def iwishrnd_fast_vector(parameter1, parameter2, num_particles):
    if parameter1.ndim == 2:
        parameter1 = parameter1[:, :, np.newaxis]
    d = np.zeros(parameter1.shape)
    d[0, 0, :] = np.sqrt(parameter1[0, 0, :])
    d[1, 0, :] = parameter1[1, 0, :] / d[0, 0, :]
    d[1, 1, :] = np.sqrt(parameter1[1, 1, :] - d[1, 0, :] ** 2)

    if parameter1.shape[2] == 1:
        d = np.tile(d, (1, 1, num_particles))

    r = 2 * np.random.gamma(
        (parameter2 - np.array([np.zeros(num_particles), np.ones(num_particles)])) / 2
    )
    x = np.sqrt(r)
    x = np.concatenate((x, np.random.randn(1, num_particles)), axis=0)

    det_x = 1 / (x[0, :] * x[1, :])
    inv_x = np.zeros((2, 2, num_particles))
    inv_x[0, 0, :] = det_x * x[1, :]
    inv_x[1, 1, :] = det_x * x[0, :]
    inv_x[0, 1, :] = -det_x * x[2, :]

    T = np.zeros((2, 2, num_particles))
    T[0, 0, :] = d[0, 0, :] * inv_x[0, 0, :] + d[0, 1, :] * inv_x[1, 0, :]
    T[0, 1, :] = d[0, 0, :] * inv_x[0, 1, :] + d[0, 1, :] * inv_x[1, 1, :]
    T[1, 0, :] = d[1, 0, :] * inv_x[0, 0, :] + d[1, 1, :] * inv_x[1, 0, :]
    T[1, 1, :] = d[1, 0, :] * inv_x[0, 1, :] + d[1, 1, :] * inv_x[1, 1, :]

    a = np.zeros((2, 2, num_particles))
    a[0, 0, :] = T[0, 0, :] * T[0, 0, :] + T[0, 1, :] * T[0, 1, :]
    a[0, 1, :] = T[0, 0, :] * T[1, 0, :] + T[0, 1, :] * T[1, 1, :]
    a[1, 0, :] = T[1, 0, :] * T[0, 0, :] + T[1, 1, :] * T[0, 1, :]
    a[1, 1, :] = T[1, 0, :] * T[1, 0, :] + T[1, 1, :] * T[1, 1, :]

    return a


def data_association_bp(input_da):
    input_da = input_da[1, :] / input_da[0, :]
    sum_input_da = 1 + np.sum(input_da)
    output_da = 1 / (sum_input_da - input_da)

    if np.any(np.isnan(output_da)):
        output_da = np.zeros(output_da.shape)
        index = np.argmax(input_da)
        output_da[index] = 1

    return output_da


def get_weights_unknown(log_weights, old_existence, skip_index):
    if skip_index:
        log_weights[:, skip_index] = 0

    log_weights = np.sum(log_weights, axis=1)
    alive_update = np.mean(np.exp(log_weights), axis=0)

    if np.isinf(alive_update):
        updated_existence = 1
    else:
        alive = old_existence * alive_update
        dead = 1 - old_existence
        updated_existence = alive / (dead + alive)

    weights = np.exp(log_weights - np.max(log_weights))
    weights = 1 / np.sum(weights) * weights

    return weights, updated_existence


def update_particles(
    old_particles_kinematic,
    old_particles_extent,
    old_existence,
    log_weights,
    parameters,
):
    num_particles = parameters["numParticles"]
    regularization_deviation = parameters["regularizationDeviation"]

    log_weights = np.sum(log_weights, axis=1)
    alive_update = np.mean(np.exp(log_weights), axis=0)

    if np.isinf(alive_update):
        updated_existence = 1
    else:
        alive = old_existence * alive_update
        dead = 1 - old_existence
        updated_existence = alive / (dead + alive)

    if updated_existence != 0:
        log_weights = log_weights - np.max(log_weights)
        weights = np.exp(log_weights)
        weights_normalized = 1 / np.sum(weights) * weights

        indexes = resample_systematic(weights_normalized, num_particles)
        updated_particles_kinematic = old_particles_kinematic[:, indexes]
        updated_particles_extent = old_particles_extent[:, :, indexes]

        updated_particles_kinematic[
            0:2, :
        ] += regularization_deviation * np.random.randn(2, num_particles)
    else:
        updated_particles_kinematic = np.full(old_particles_kinematic.shape, np.nan)
        updated_particles_extent = np.full(old_particles_extent.shape, np.nan)

    return updated_particles_kinematic, updated_particles_extent, updated_existence


def track_formation(estimates, parameters):
    num_steps = len(estimates)
    minimum_track_length = parameters["minimumTrackLength"]

    labels = np.zeros((2, 0))
    for step in range(num_steps):
        if estimates[step] is not None:
            current_labels = estimates[step]["label"]
            labels = np.concatenate((labels, current_labels), axis=1)

    labels = np.unique(labels, axis=1)
    _, num_tracks = labels.shape
    tracks = np.full((4, num_steps, num_tracks), np.nan)
    extents = np.full((2, 2, num_steps, num_tracks), np.nan)

    for step in range(num_steps):
        if estimates[step] is not None:
            current_labels = estimates[step]["label"]
            current_states = estimates[step]["state"]
            current_extents = estimates[step]["extent"]
            _, num_estimates = current_labels.shape

            for estimate in range(num_estimates):
                indexes = compare_vector_with_matrix(
                    labels, current_labels[:, estimate]
                )
                tracks[:, step, indexes] = current_states[:, estimate]
                extents[:, :, step, indexes] = current_extents[:, :, estimate]

    indexes = np.zeros(num_tracks, dtype=bool)
    for track in range(num_tracks):
        if np.sum(~np.isnan(tracks[0, :, track])) >= minimum_track_length:
            indexes[track] = True

    tracks = tracks[:, :, indexes]
    extents = extents[:, :, :, indexes]

    return tracks, extents


def compare_vector_with_matrix(matrix, vector):
    return np.where(np.all(matrix == vector[:, np.newaxis], axis=0))[0]


def get_transition_matrices(scan_time):
    A = np.diag(np.ones(4))
    A[0, 2] = scan_time
    A[1, 3] = scan_time

    W = np.zeros((4, 2))
    W[0, 0] = 0.5 * scan_time**2
    W[1, 1] = 0.5 * scan_time**2
    W[2, 0] = scan_time
    W[3, 1] = scan_time

    return A, W


def get_central_reordered(measurements, probabilities_new, parameters):
    parameters["freeThreshold"] = 0.85
    parameters["clusterThreshold"] = 0.9
    parameters["minClusterElements"] = 2

    threshold = parameters["freeThreshold"]
    cluster_threshold = parameters["clusterThreshold"]
    mean_extent_birth = (
        parameters["priorExtent1"] / (parameters["priorExtent2"] - 3)
    ) ** 2
    measurements_covariance = (
        parameters["measurementVariance"] * np.eye(2) + mean_extent_birth
    )
    min_cluster_elements = parameters["minClusterElements"]

    all_indexes_numeric = np.arange(measurements.shape[1])
    free_indexes = probabilities_new >= threshold
    assigned_indexes = probabilities_new < threshold

    measurements_free = measurements[:, free_indexes.flatten()]
    free_indexes_numeric = all_indexes_numeric[free_indexes.flatten()]
    assigned_indexes_numeric = all_indexes_numeric[assigned_indexes.flatten()]

    clusters = get_clusters(
        measurements_free, measurements_covariance, cluster_threshold
    )
    if clusters.ndim == 1:
        clusters = clusters[:, np.newaxis]

    num_elements = np.sum(clusters > 0, axis=0)
    indexes = np.argsort(num_elements)[::-1]
    num_elements = num_elements[indexes]
    clusters = clusters[:, indexes]

    not_used_indexes = clusters[:, num_elements < min_cluster_elements]
    not_used_indexes = not_used_indexes[not_used_indexes != 0].astype(int)
    not_used_indexes_numeric = free_indexes_numeric[not_used_indexes]
    num_not_used = not_used_indexes_numeric.shape[0]

    clusters = clusters[:, num_elements >= min_cluster_elements]
    indexes_numeric_new = np.zeros(0, dtype=int)
    num_clusters = clusters.shape[1]
    central_indexes = np.zeros(num_clusters, dtype=int)

    for cluster in range(num_clusters):
        indexes = clusters[:, cluster][clusters[:, cluster] != 0]
        current_measurements = measurements_free[:, indexes]
        current_indexes_numeric = free_indexes_numeric[indexes]

        if len(indexes) > 1:
            num_measurements = len(indexes)
            distance_matrix = np.zeros((num_measurements, num_measurements))
            for m1 in range(num_measurements):
                for m2 in range(m1 + 1, num_measurements):
                    dist_vector = (
                        current_measurements[:, m1] - current_measurements[:, m2]
                    )
                    distance_matrix[m1, m2] = np.sqrt(
                        dist_vector.T
                        @ np.linalg.inv(measurements_covariance)
                        @ dist_vector
                    )
                    distance_matrix[m2, m1] = distance_matrix[m1, m2]

            distance_vector = np.sum(distance_matrix, axis=1)
            indexes = np.argsort(distance_vector)[::-1]
            current_indexes_numeric = current_indexes_numeric[indexes]

        indexes_numeric_new = np.concatenate(
            (current_indexes_numeric, indexes_numeric_new)
        )
        central_indexes[0 : cluster + 1] = central_indexes[0 : cluster + 1] + len(
            indexes
        )

    indexes_reordered = np.concatenate(
        (not_used_indexes_numeric, indexes_numeric_new, assigned_indexes_numeric)
    )
    central_indexes = central_indexes + num_not_used
    central_indexes = np.sort(central_indexes)[::-1]

    return central_indexes, indexes_reordered


def get_clusters(measurements, measurements_covariance, threshold_probability):
    num_measurements = measurements.shape[1]
    if not num_measurements:
        return np.array([])

    threshold_distance = chi2.ppf(threshold_probability, 2)
    distance_vector = np.zeros(int(num_measurements * (num_measurements - 1) / 2 + 1))
    distance_matrix = np.zeros((num_measurements, num_measurements))
    entry = 0

    for m1 in range(num_measurements):
        for m2 in range(m1 + 1, num_measurements):
            dist_vector = measurements[:, m1] - measurements[:, m2]
            entry += 1
            distance_vector[entry] = np.sqrt(
                dist_vector.T @ np.linalg.inv(measurements_covariance) @ dist_vector
            )
            distance_matrix[m1, m2] = distance_vector[entry]
            distance_matrix[m2, m1] = distance_vector[entry]

    distance_vector = np.sort(distance_vector)
    distance_vector = distance_vector[distance_vector <= threshold_distance]
    distance = distance_vector[-1]

    cluster_numbers = np.zeros(num_measurements, dtype=int)
    cluster_id = 1
    for m in range(num_measurements):
        if cluster_numbers[m] == 0:
            cluster_numbers[m] = cluster_id
            cluster_numbers = find_neighbors(
                m, cluster_numbers, cluster_id, distance_matrix, distance
            )
            cluster_id += 1

    num_clusters = cluster_id - 1
    max_elements = np.sum(cluster_numbers == np.argmax(np.bincount(cluster_numbers)))
    clusters = np.zeros((0, max_elements), dtype=int)
    index = 0

    for c in range(1, num_clusters + 1):
        association_tmp = np.where(cluster_numbers == c)[0]
        num_elements = len(association_tmp)
        if num_elements <= max_elements:
            index += 1
            clusters = np.vstack(
                (
                    clusters,
                    np.pad(
                        association_tmp, (max_elements - num_elements, 0), "constant"
                    ),
                )
            )

    return clusters


def find_neighbors(index, cell_numbers, cell_id, distance_matrix, distance_threshold):
    num_measurements = distance_matrix.shape[1]
    for m in range(num_measurements):
        if (
            m != index
            and distance_matrix[m, index] < distance_threshold
            and cell_numbers[m] == 0
        ):
            cell_numbers[m] = cell_id
            cell_numbers = find_neighbors(
                index, cell_numbers, cell_id, distance_matrix, distance_threshold
            )
    return cell_numbers


def iwishrnd(S, df):
    # This is a placeholder. The actual implementation requires a more complex statistical calculation.
    return np.diag(np.random.rand(2) * 5 + 2)


def resample_systematic(weights, num_samples):
    # This is a placeholder.
    weights[np.isnan(weights)] = 0
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        weights = np.ones(len(weights)) / len(weights)
    return np.random.choice(len(weights), size=num_samples, p=weights)


def mvnpdf(x, mu, sigma):
    # This is a placeholder.
    return 1.0
