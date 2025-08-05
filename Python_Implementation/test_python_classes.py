#!/usr/bin/env python3
"""
Test script for Python Extended Object Tracking classes.
Demonstrates usage of DataGenerator and ExtendedObjectFilter classes.
"""

import numpy as np
import matplotlib.pyplot as plt
from DataGenerator import DataGenerator
from ExtendedObjectFilter import ExtendedObjectFilter
from utils import generate_default_parameters, set_prior_extent, show_results

def main():
    """
    Main test function replicating MATLAB main.m
    """
    # Set random seed for reproducibility
    np.random.seed(1)
    
    # Simulation parameters (matching main.m)
    num_steps = 50
    num_targets = 5
    mean_target_dimension = 3
    start_radius = 75
    start_velocity = 10
    
    # Target appearance intervals (matching main.m)
    appearance_from_to = np.array([
        [3, 83], [3, 83], [6, 86], [6, 86], [9, 89], 
        [9, 89], [12, 92], [12, 92], [15, 95], [15, 95]
    ]).T  # Shape: (2, 10) but we'll use first 5 columns
    appearance_from_to = appearance_from_to[:, :num_targets]
    
    print("Setting up parameters...")
    
    # Generate default parameters
    parameters = generate_default_parameters()
    parameters = set_prior_extent(parameters, mean_target_dimension)
    
    print(f"Simulating {num_targets} targets over {num_steps} time steps...")
    
    # Initialize data generator
    data_gen = DataGenerator(parameters)
    
    # Generate simulation data
    target_tracks, target_extents, measurements = data_gen.simulate_scenario(
        num_steps=num_steps,
        num_targets=num_targets,
        mean_target_dimension=mean_target_dimension,
        start_radius=start_radius,
        start_velocity=start_velocity,
        appearance_from_to=appearance_from_to
    )
    
    print(f"Generated {len(measurements)} measurement sets")
    total_measurements = sum(m.shape[1] for m in measurements)
    print(f"Total measurements: {total_measurements}")
    
    print("Running Extended Object Filter...")
    
    # Initialize and run filter
    eot_filter = ExtendedObjectFilter(parameters)
    estimated_tracks, estimated_extents = eot_filter.run_filter(measurements)
    
    print(f"Filter completed. Estimated {estimated_tracks.shape[2]} tracks")
    
    # Display results
    print("\nTrue tracks summary:")
    for i in range(num_targets):
        valid_steps = ~np.isnan(target_tracks[0, :, i])
        print(f"  Target {i+1}: {np.sum(valid_steps)} valid time steps")
        
    print("\nEstimated tracks summary:")
    for i in range(estimated_tracks.shape[2]):
        valid_steps = ~np.isnan(estimated_tracks[0, :, i])
        print(f"  Track {i+1}: {np.sum(valid_steps)} valid time steps")
    
    # Visualization
    print("\nDisplaying results...")
    axis_limits = [-150, 150, -150, 150]
    
    try:
        show_results(
            target_tracks=target_tracks,
            target_extents=target_extents,
            estimated_tracks=estimated_tracks,
            estimated_extents=estimated_extents,
            measurements=measurements,
            axis_limits=axis_limits,
            mode=0  # Final result mode
        )
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Skipping visualization...")
    
    # Compute basic performance metrics
    print("\nPerformance Summary:")
    print(f"Number of true targets: {num_targets}")
    print(f"Number of estimated tracks: {estimated_tracks.shape[2]}")
    
    # Simple position error for first target/track (if available)
    if (num_targets > 0 and estimated_tracks.shape[2] > 0):
        true_pos = target_tracks[:2, :, 0]
        est_pos = estimated_tracks[:2, :, 0]
        
        # Find overlapping valid time steps
        true_valid = ~np.isnan(true_pos[0, :])
        est_valid = ~np.isnan(est_pos[0, :])
        common_valid = true_valid & est_valid
        
        if np.any(common_valid):
            pos_errors = np.linalg.norm(
                true_pos[:, common_valid] - est_pos[:, common_valid], axis=0
            )
            print(f"Mean position error (first target): {np.mean(pos_errors):.2f}")
            print(f"RMS position error (first target): {np.sqrt(np.mean(pos_errors**2)):.2f}")
        else:
            print("No overlapping valid time steps for error computation")
    
    print("\nTest completed successfully!")
    
    return target_tracks, target_extents, measurements, estimated_tracks, estimated_extents

def test_data_generator_only():
    """
    Test only the data generation functionality
    """
    print("Testing DataGenerator class only...")
    
    # Generate parameters
    parameters = generate_default_parameters()
    parameters = set_prior_extent(parameters, 3.0)
    
    # Create data generator
    data_gen = DataGenerator(parameters)
    
    # Test individual methods
    print("Testing get_transition_matrices...")
    A, W, rA, rW = data_gen.get_transition_matrices(0.2)
    print(f"A shape: {A.shape}, W shape: {W.shape}")
    
    print("Testing get_start_states...")
    start_states, start_matrices = data_gen.get_start_states(3, 50, 5)
    print(f"Start states shape: {start_states.shape}")
    print(f"Start matrices shape: {start_matrices.shape}")
    
    print("Testing generate_tracks_unknown...")
    appearance = np.array([[1, 20], [1, 20], [1, 20]]).T
    tracks, extents = data_gen.generate_tracks_unknown(
        start_states, start_matrices, appearance, 25
    )
    print(f"Tracks shape: {tracks.shape}")
    print(f"Extents shape: {extents.shape}")
    
    print("Testing generate_cluttered_measurements...")
    measurements = data_gen.generate_cluttered_measurements(tracks, extents)
    print(f"Number of measurement sets: {len(measurements)}")
    
    print("DataGenerator test completed!")

if __name__ == "__main__":
    print("Extended Object Tracking Python Implementation Test")
    print("=" * 50)
    
    # Test choice
    test_choice = input("Enter test choice (1=Full test, 2=DataGenerator only, 3=Both): ")
    
    if test_choice == "1":
        main()
    elif test_choice == "2":
        test_data_generator_only()
    elif test_choice == "3":
        test_data_generator_only()
        print("\n" + "=" * 50)
        main()
    else:
        print("Running full test by default...")
        main()