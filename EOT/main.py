import numpy as np
from eot_filter import EOTFilter
import _common as common


import Util

def main():
    # parameters of simulated scenario
    num_steps = 50
    num_targets = 5
    start_radius = 75
    start_velocity = 10

    parameters = Util.get_parameters()

    # generate true start states
    start_states, start_matrixes = common.get_start_states(
        num_targets, start_radius, start_velocity, parameters
    )
    appearance_from_to = np.array(
        [
            [3, 83],
            [3, 83],
            [6, 86],
            [6, 86],
            [9, 89],
            [9, 89],
            [12, 92],
            [12, 92],
            [15, 95],
            [15, 95],
        ]
    )

    # generate true track
    target_tracks, target_extents = common.generate_tracks_unknown(
        parameters, start_states, start_matrixes, appearance_from_to, num_steps
    )

    # generate measurements
    measurements = common.generate_cluttered_measurements(
        target_tracks, target_extents, parameters
    )

    # perform graph-based extended object tracking (EOT)
    eot_filter = EOTFilter(parameters)
    estimated_tracks, estimated_extents = eot_filter.filter(measurements)

    # show results
    # mode = 2  # hit ``space'' to start visualization; set mode=0 for final result and mode=2 to frame-by-frame.
    # common.show_results(
    #     target_tracks,
    #     target_extents,
    #     estimated_tracks,
    #     estimated_extents,
    #     measurements,
    #     np.array([-150, 150, -150, 150]),
    #     mode,
    # )


if __name__ == "__main__":
    main()
