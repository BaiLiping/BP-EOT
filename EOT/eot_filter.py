import numpy as np
import _common as common


class EOTFilter:
    def __init__(self, parameters):
        self.parameters = parameters

    def filter(self, measurements_cell):
        num_particles = self.parameters["numParticles"]
        mean_clutter = self.parameters["meanClutter"]
        mean_measurements = self.parameters["meanMeasurements"]
        scan_time = self.parameters["scanTime"]
        detection_threshold = self.parameters["detectionThreshold"]
        threshold_pruning = self.parameters["thresholdPruning"]
        num_outer_iterations = self.parameters["numOuterIterations"]
        mean_births = self.parameters["meanBirths"]
        prior_velocity_covariance = self.parameters["priorVelocityCovariance"]
        surveillance_region = self.parameters["surveillanceRegion"]

        area_size = (surveillance_region[1, 0] - surveillance_region[0, 0]) * (
            surveillance_region[1, 1] - surveillance_region[0, 1]
        )
        measurements_covariance = self.parameters["measurementVariance"] * np.eye(2)

        prior_extent1 = self.parameters["priorExtent1"]
        prior_extent2 = self.parameters["priorExtent2"]
        mean_extent_prior = np.dot(prior_extent1, np.linalg.inv(prior_extent1)) / (
            prior_extent2 - 3
        )
        total_covariance = (
            np.dot(mean_extent_prior, mean_extent_prior) + measurements_covariance
        )

        num_steps = len(measurements_cell)
        constant_factor = area_size * (mean_measurements / mean_clutter)
        uniform_weight = np.log(1 / area_size)

        estimates = [None] * num_steps
        current_labels = np.zeros((2, 0))
        current_particles_kinematic = np.zeros((4, num_particles, 0))
        current_existences = np.zeros((0, 1))
        current_particles_extent = np.zeros((2, 2, num_particles, 0))

        for step in range(num_steps):
            print(f"Step {step + 1}")

            measurements = measurements_cell[step]
            num_measurements = measurements.shape[1]

            (
                current_particles_kinematic,
                current_existences,
                current_particles_extent,
            ) = common.perform_prediction(
                current_particles_kinematic,
                current_existences,
                current_particles_extent,
                scan_time,
                self.parameters,
            )

            current_alive = current_existences * np.exp(-mean_measurements)
            current_dead = 1 - current_existences
            current_existences = current_alive / (current_dead + current_alive)
            num_targets = current_particles_kinematic.shape[2]
            num_legacy = num_targets

            new_indexes, measurements = common.get_promising_new_targets(
                current_particles_kinematic,
                current_particles_extent,
                current_existences,
                measurements,
                self.parameters,
            )
            if measurements.shape[1] == 0:
                continue
            if new_indexes.shape[0] == 0:
                continue
            num_new = new_indexes.shape[0]
            current_labels = np.concatenate(
                (
                    current_labels,
                    np.vstack(((step + 1) * np.ones(num_new), new_indexes)),
                ),
                axis=1,
            )

            new_existences = np.tile(
                mean_births
                * np.exp(-mean_measurements)
                / (mean_births * np.exp(-mean_measurements) + 1),
                (num_new, 1),
            )
            new_particles_kinematic = np.zeros((4, num_particles, num_new))
            new_particles_extent = np.zeros((2, 2, num_particles, num_new))
            new_weights = np.zeros((num_particles, num_new))

            for target in range(num_new):
                if new_indexes[target] >= measurements.shape[1]:
                    new_indexes[target] = measurements.shape[1] - 1
                proposal_mean = measurements[:, new_indexes[target]]
                proposal_covariance = 2 * total_covariance

                new_particles_kinematic[0:2, :, target] = proposal_mean[
                    :, np.newaxis
                ] + np.linalg.cholesky(proposal_covariance) @ np.random.randn(
                    2, num_particles
                )
                new_weights[:, target] = uniform_weight - np.log(
                    common.mvnpdf(
                        new_particles_kinematic[0:2, :, target].T,
                        proposal_mean,
                        proposal_covariance,
                    )
                )

                new_particles_extent[
                    :, :, :, target
                ] = common.iwishrnd_fast_vector(
                    prior_extent1, prior_extent2, num_particles
                )

            current_existences = np.concatenate(
                (current_existences, new_existences), axis=0
            )
            current_existences_extrinsic = np.tile(
                current_existences, (1, num_measurements)
            )

            current_particles_kinematic = np.concatenate(
                (current_particles_kinematic, new_particles_kinematic), axis=2
            )
            current_particles_extent = np.concatenate(
                (current_particles_extent, new_particles_extent), axis=3
            )

            weights_extrinsic = np.full(
                (num_particles, num_measurements, num_legacy), np.nan
            )
            weights_extrinsic_new = np.full(
                (num_particles, num_measurements, new_indexes.shape[0]), np.nan
            )

            likelihood1 = np.zeros((num_particles, num_measurements, num_targets))
            likelihood_new1 = np.full(
                (num_particles, num_measurements, new_indexes.shape[0]), np.nan
            )

            for outer in range(num_outer_iterations):
                output_da = [None] * num_measurements
                target_indexes = [None] * num_measurements

                for measurement in range(num_measurements - 1, -1, -1):
                    input_da = np.ones((2, num_legacy))

                    for target in range(num_legacy):
                        if outer == 0:
                            if measurement >= measurements.shape[1]:
                                measurement = measurements.shape[1] -1
                            likelihood1[:, measurement, target] = (
                                constant_factor
                                * np.exp(
                                    common.get_log_weights_fast(
                                        measurements[:, measurement],
                                        current_particles_kinematic[0:2, :, target],
                                        common.get_square2_fast(
                                            current_particles_extent[:, :, :, target]
                                        )
                                        + measurements_covariance[:,:,np.newaxis],
                                    )
                                )
                            )
                            input_da[1, target] = current_existences_extrinsic[
                                target, measurement
                            ] * np.mean(likelihood1[:, measurement, target], axis=0)
                        else:
                            input_da[1, target] = current_existences_extrinsic[
                                target, measurement
                            ] * (
                                weights_extrinsic[:, measurement, target].T
                                @ likelihood1[:, measurement, target]
                            )

                        input_da[0, target] = 1

                    target_index = num_legacy
                    target_indexes_current = np.full(num_legacy, np.nan)

                    for target in range(num_measurements - 1, measurement - 1, -1):
                        if np.any(target == new_indexes):
                            target_index += 1
                            target_indexes_current = np.concatenate(
                                (target_indexes_current, [target])
                            )

                            if outer == 0:
                                weights = np.exp(
                                    new_weights[:, target_index - num_legacy - 1]
                                )
                                weights = (weights / np.sum(weights, axis=0)).T
                                likelihood_new1[
                                    :, measurement, target_index - num_legacy - 1
                                ] = constant_factor * np.exp(
                                    common.get_log_weights_fast(
                                        measurements[:, measurement],
                                        current_particles_kinematic[
                                            0:2, :, target_index - 1
                                        ],
                                        common.get_square2_fast(
                                            current_particles_extent[
                                                :, :, :, target_index - 1
                                            ]
                                        )
                                        + measurements_covariance[:,:,np.newaxis],
                                    )
                                )
                                if likelihood_new1.shape[2] > target_index - num_legacy - 1 and input_da.shape[1] > target_index-1:
                                    input_da[
                                        1, target_index - 1
                                    ] = current_existences_extrinsic[
                                        target_index - 1, measurement
                                    ] * (
                                        weights
                                        @ likelihood_new1[
                                            :, measurement, target_index - num_legacy - 1
                                        ]
                                    )
                            elif input_da.shape[1] > target_index -1:
                                input_da[
                                    1, target_index - 1
                                ] = current_existences_extrinsic[
                                    target_index - 1, measurement
                                ] * (
                                    weights_extrinsic_new[
                                        :, measurement, target_index - num_legacy - 1
                                    ].T
                                    @ likelihood_new1[
                                        :, measurement, target_index - num_legacy - 1
                                    ]
                                )
                            if input_da.shape[1] > target_index - 1:
                                input_da[0, target_index - 1] = 1

                                if target == measurement:
                                    input_da[0, target_index - 1] = (
                                        1
                                        - current_existences_extrinsic[
                                            target_index - 1, measurement
                                        ]
                                    )

                    target_indexes[measurement] = target_indexes_current
                    output_da[measurement] = common.data_association_bp(input_da)

                for target in range(num_legacy):
                    weights = np.zeros(
                        (current_particles_kinematic.shape[1], num_measurements)
                    )
                    for measurement in range(num_measurements):
                        if output_da[measurement] is not None:
                            current_weights = (
                                1
                                + likelihood1[:, measurement, target]
                                * output_da[measurement][target]
                            )
                        current_weights = np.log(current_weights)
                        weights[:, measurement] = current_weights

                    if outer != num_outer_iterations - 1:
                        for measurement in range(num_measurements):
                            (
                                weights_extrinsic[:, measurement, target],
                                current_existences_extrinsic[target, measurement],
                            ) = common.get_weights_unknown(
                                weights, current_existences[target], measurement
                            )
                    else:
                        (
                            current_particles_kinematic[:, :, target],
                            current_particles_extent[:, :, :, target],
                            current_existences[target],
                        ) = common.update_particles(
                            current_particles_kinematic[:, :, target],
                            current_particles_extent[:, :, :, target],
                            current_existences[target],
                            weights,
                            self.parameters,
                        )

                target_index = num_legacy
                for target in range(num_measurements - 1, -1, -1):
                    if np.any(target == new_indexes):
                        target_index += 1
                        weights = np.zeros(
                            (current_particles_kinematic.shape[1], num_measurements + 1)
                        )
                        weights[:, num_measurements] = new_weights[
                            :, target_index - num_legacy - 1
                        ]

                        for measurement in range(target + 1):
                            if target_indexes[measurement] is not None and len(target_indexes[measurement]) > 0 and not np.all(np.isnan(target_indexes[measurement])) and len(output_da[measurement]) > 0:
                                bool_idx = target_indexes[measurement] == target
                                if bool_idx.shape[0] == output_da[measurement].shape[0]:
                                    output_tmp_da = output_da[measurement][bool_idx]
                                else:
                                    output_tmp_da = output_da[measurement]

                                if not np.any(np.isinf(output_tmp_da)) and len(output_tmp_da) > 0:
                                    current_weights = (
                                        likelihood_new1[
                                            :, measurement, target_index - num_legacy - 1
                                        ]
                                        * output_tmp_da
                                    )
                                else:
                                    current_weights = likelihood_new1[
                                        :, measurement, target_index - num_legacy - 1
                                    ]

                                if measurement != target:
                                    current_weights = current_weights + 1

                                current_weights = np.log(current_weights)
                                weights[:, measurement] = current_weights

                        if outer != num_outer_iterations - 1:
                            for measurement in range(target + 1):
                                (
                                    weights_extrinsic_new[
                                        :, measurement, target_index - num_legacy - 1
                                    ],
                                    current_existences_extrinsic[
                                        target_index - 1, measurement
                                    ],
                                ) = common.get_weights_unknown(
                                    weights,
                                    current_existences[target_index - 1],
                                    measurement,
                                )
                        else:
                            (
                                current_particles_kinematic[0:2, :, target_index - 1],
                                current_particles_extent[:, :, :, target_index - 1],
                                current_existences[target_index - 1],
                            ) = common.update_particles(
                                current_particles_kinematic[0:2, :, target_index - 1],
                                current_particles_extent[:, :, :, target_index - 1],
                                current_existences[target_index - 1],
                                weights,
                                self.parameters,
                            )
                            current_particles_kinematic[2:4, :, target_index - 1] = (
                                np.random.multivariate_normal(
                                    np.zeros(2),
                                    prior_velocity_covariance,
                                    num_particles,
                                ).T
                            )

            num_targets = current_particles_kinematic.shape[2]
            is_redundant = np.zeros(num_targets, dtype=bool)
            for target in range(num_targets):
                if current_existences[target] < threshold_pruning:
                    is_redundant[target] = True

            current_particles_kinematic = current_particles_kinematic[
                :, :, ~is_redundant
            ]
            current_particles_extent = current_particles_extent[:, :, :, ~is_redundant]
            current_labels = current_labels[:, ~is_redundant]
            current_existences = current_existences[~is_redundant]

            num_targets = current_particles_kinematic.shape[2]
            detected_targets = 0
            for target in range(num_targets):
                if current_existences[target] > detection_threshold:
                    detected_targets += 1
                    if estimates[step] is None:
                        estimates[step] = {
                            "state": np.zeros((4, 0)),
                            "extent": np.zeros((2, 2, 0)),
                            "label": np.zeros((2, 0)),
                        }
                    estimates[step]["state"] = np.concatenate(
                        (
                            estimates[step]["state"],
                            np.mean(current_particles_kinematic[:, :, target], axis=1)[
                                :, np.newaxis
                            ],
                        ),
                        axis=1,
                    )
                    estimates[step]["extent"] = np.concatenate(
                        (
                            estimates[step]["extent"],
                            np.mean(current_particles_extent[:, :, :, target], axis=2)[
                                :, :, np.newaxis
                            ],
                        ),
                        axis=2,
                    )
                    estimates[step]["label"] = np.concatenate(
                        (
                            estimates[step]["label"],
                            current_labels[:, target][:, np.newaxis],
                        ),
                        axis=1,
                    )

        estimated_tracks, estimated_extents = common.track_formation(
            estimates, self.parameters
        )
        return estimated_tracks, estimated_extents
