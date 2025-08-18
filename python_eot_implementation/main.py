#!/usr/bin/env python3
"""
Extended Object Tracking with Elliptical Shapes

This is a Python implementation of the Extended Object Tracking (EOT) algorithm
described in:

F. Meyer and J. L. Williams, "Scalable detection and tracking of geometric extended objects,"
IEEE Trans. Signal Process., vol. 69, pp. 6283–6298, Oct. 2021.

Original MATLAB implementation by Florian Meyer, 2020
Python conversion with detailed comments for educational purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

# Import our common utilities
from eot_common import (
    get_start_states, generate_tracks_unknown, generate_cluttered_measurements,
    show_results, EOTParameters
)
from eot_elliptical_shape import eot_elliptical_shape


def main():
    """
    Main function that runs the Extended Object Tracking simulation.
    
    This function:
    1. Sets up simulation parameters for targets and tracking
    2. Generates ground truth target trajectories
    3. Simulates noisy measurements with clutter
    4. Runs the EOT algorithm to estimate target states
    5. Visualizes the results
    """
    
    # Set random seed for reproducibility
    np.random.seed(1)
    
    # ==========================================================================
    # SIMULATION SCENARIO PARAMETERS
    # ==========================================================================
    
    # Number of time steps in the simulation
    num_steps = 50
    
    # Number of targets to track
    num_targets = 5
    
    # Mean dimension of target extents (used for initialization)
    mean_target_dimension = 3
    
    # Radius at which targets start (circular formation)
    start_radius = 75
    
    # Initial velocity magnitude of targets
    start_velocity = 10
    
    
    # ==========================================================================
    # STATISTICAL MODEL PARAMETERS
    # ==========================================================================
    
    parameters = EOTParameters()
    
    # Time between scans (seconds)
    parameters.scan_time = 0.2
    
    # Standard deviation of acceleration noise in target dynamics
    parameters.acceleration_deviation = 1
    
    # Probability that a target survives from one time step to the next
    parameters.survival_probability = 0.99
    
    # Expected number of new targets appearing per time step
    parameters.mean_births = 0.01
    
    # Surveillance region boundaries [x_min, x_max; y_min, y_max]
    parameters.surveillance_region = np.array([[-200, 200], [-200, 200]])
    
    # Variance of measurement noise
    parameters.measurement_variance = 1.0**2
    
    # Expected number of measurements per target per time step
    parameters.mean_measurements = 8
    
    # Expected number of false alarm (clutter) measurements per time step
    parameters.mean_clutter = 10
    
    
    # ==========================================================================
    # PRIOR DISTRIBUTION PARAMETERS
    # ==========================================================================
    
    # Prior covariance for target velocity components
    parameters.prior_velocity_covariance = np.diag([10**2, 10**2])
    
    # Parameters for inverse Wishart prior on target extent
    # These define the shape and scale of the prior distribution
    parameters.prior_extent_2 = 100  # Degrees of freedom
    parameters.prior_extent_1 = (
        np.eye(2) * mean_target_dimension * (parameters.prior_extent_2 - 3)
    )
    
    # Degrees of freedom for extent prediction (large value for stability)
    parameters.degree_freedom_prediction = 20000
    
    
    # ==========================================================================
    # PARTICLE FILTER PARAMETERS  
    # ==========================================================================
    
    # Number of particles used in the particle filter
    parameters.num_particles = 5000
    
    # Standard deviation for particle regularization (0 = no regularization)
    parameters.regularization_deviation = 0
    
    
    # ==========================================================================
    # DETECTION AND PRUNING PARAMETERS
    # ==========================================================================
    
    # Threshold for declaring a track as detected
    parameters.detection_threshold = 0.5
    
    # Threshold for pruning low-probability tracks
    parameters.threshold_pruning = 1e-3
    
    # Minimum track length before a track can be output
    parameters.minimum_track_length = 1
    
    
    # ==========================================================================
    # MESSAGE PASSING PARAMETERS
    # ==========================================================================
    
    # Number of belief propagation iterations per time step
    parameters.num_outer_iterations = 2
    
    
    # ==========================================================================
    # GENERATE GROUND TRUTH SCENARIO
    # ==========================================================================
    
    print("Generating ground truth target states and trajectories...")
    
    # Generate initial target states and extent matrices
    start_states, start_matrices = get_start_states(
        num_targets, start_radius, start_velocity, parameters
    )
    
    # Define appearance and disappearance times for each target
    # Format: [appearance_time, disappearance_time] for each target
    appearance_from_to = np.array([
        [3, 83], [3, 83], [6, 86], [6, 86], [9, 89],
        [9, 89], [12, 92], [12, 92], [15, 95], [15, 95]
    ]).T  # Transpose to match MATLAB indexing
    
    # Generate true target trajectories
    target_tracks, target_extents = generate_tracks_unknown(
        parameters, start_states, start_matrices, appearance_from_to, num_steps
    )
    
    # Generate noisy measurements with clutter
    measurements = generate_cluttered_measurements(
        target_tracks, target_extents, parameters
    )
    
    
    # ==========================================================================
    # RUN EXTENDED OBJECT TRACKING ALGORITHM
    # ==========================================================================
    
    print("Running Extended Object Tracking algorithm...")
    start_time = time.time()
    
    # Run the main EOT algorithm
    estimated_tracks, estimated_extents = eot_elliptical_shape(measurements, parameters)
    
    end_time = time.time()
    print(f"EOT algorithm completed in {end_time - start_time:.2f} seconds")
    
    
    # ==========================================================================
    # VISUALIZE RESULTS
    # ==========================================================================
    
    print("Displaying results...")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EXTENDED OBJECT TRACKING RESULTS SUMMARY")
    print("="*60)
    
    print(f"\nSimulation Parameters:")
    print(f"  - Number of time steps: {num_steps}")
    print(f"  - Number of true targets: {num_targets}")
    print(f"  - Number of particles: {parameters.num_particles}")
    print(f"  - Mean measurements per target: {parameters.mean_measurements}")
    print(f"  - Mean clutter per step: {parameters.mean_clutter}")
    
    print(f"\nTrue Targets:")
    for i in range(num_targets):
        valid_steps = np.sum(~np.isnan(target_tracks[0, :, i]))
        print(f"  - Target {i+1}: Active for {valid_steps} time steps")
    
    print(f"\nEstimated Tracks:")
    if estimated_tracks.shape[2] > 0:
        for i in range(estimated_tracks.shape[2]):
            valid_steps = np.sum(~np.isnan(estimated_tracks[0, :, i]))
            print(f"  - Track {i+1}: {valid_steps} detections")
    else:
        print("  - No tracks detected")
    
    print(f"\nMeasurements per time step:")
    total_measurements = sum(meas.shape[1] for meas in measurements)
    avg_measurements = total_measurements / num_steps
    print(f"  - Total measurements: {total_measurements}")
    print(f"  - Average per time step: {avg_measurements:.1f}")
    
    # Visualization mode:
    # 0 = show final result only
    # 1 = automatic frame-by-frame animation  
    # 2 = manual frame-by-frame (press space to advance)
    mode = 0  # Use mode 0 for non-interactive display
    
    # Define plot limits [x_min, x_max, y_min, y_max]
    plot_limits = [-150, 150, -150, 150]
    
    # Display tracking results
    show_results(
        target_tracks, target_extents,
        estimated_tracks, estimated_extents, 
        measurements, plot_limits, mode
    )
    
    print("\nEOT algorithm completed successfully!")
    print("Note: Visualization disabled in command-line mode.")


if __name__ == "__main__":
    main()