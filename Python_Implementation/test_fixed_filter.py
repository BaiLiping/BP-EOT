#!/usr/bin/env python3
"""
Test script for the fixed Extended Object Filter implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from DataGenerator import DataGenerator
from ExtendedObjectFilterFixed import ExtendedObjectFilterFixed
from utils import generate_default_parameters, set_prior_extent, show_results

def test_fixed_filter():
    """
    Test the fixed Extended Object Filter with proper new target detection.
    """
    print("Testing Fixed Extended Object Filter")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(1)
    
    # Parameters from main.m
    num_steps = 50
    num_targets = 5
    mean_target_dimension = 3
    start_radius = 75
    start_velocity = 10
    
    # Setup parameters exactly matching MATLAB
    parameters = generate_default_parameters()
    parameters = set_prior_extent(parameters, mean_target_dimension)
    
    # Fix surveillance region format if needed
    parameters['surveillanceRegion'] = np.array([[-200, -200], [200, 200]])
    
    print(f"Scenario parameters:")
    print(f"  Number of steps: {num_steps}")
    print(f"  Number of targets: {num_targets}")
    print(f"  Mean target dimension: {mean_target_dimension}")
    print(f"  Number of particles: {parameters['numParticles']}")
    
    # Appearance intervals from main.m
    appearance_from_to = np.array([
        [3, 83], [3, 83], [6, 86], [6, 86], [9, 89]
    ]).T
    
    # Limit to actual number of steps
    appearance_from_to[1, :] = np.minimum(appearance_from_to[1, :], num_steps)
    
    # Generate data
    print("\nGenerating simulation data...")
    data_gen = DataGenerator(parameters)
    
    # Generate start states
    start_states, start_matrices = data_gen.get_start_states(
        num_targets, start_radius, start_velocity
    )
    
    # Generate tracks
    target_tracks, target_extents = data_gen.generate_tracks_unknown(
        start_states, start_matrices, appearance_from_to, num_steps
    )
    
    # Generate measurements
    measurements = data_gen.generate_cluttered_measurements(target_tracks, target_extents)
    
    total_measurements = sum(m.shape[1] for m in measurements if m.shape[1] > 0)
    print(f"Generated {len(measurements)} measurement sets")
    print(f"Total measurements: {total_measurements}")
    
    # Count true target appearances
    for i in range(num_targets):
        valid_steps = ~np.isnan(target_tracks[0, :, i])
        print(f"  True target {i+1}: {np.sum(valid_steps)} time steps")
    
    # Run fixed filter
    print("\nRunning Fixed Extended Object Filter...")
    start_time = time.time()
    
    eot_filter = ExtendedObjectFilterFixed(parameters)
    estimated_tracks, estimated_extents = eot_filter.run_filter(measurements)
    
    elapsed_time = time.time() - start_time
    print(f"\nFilter completed in {elapsed_time:.2f} seconds")
    print(f"Detected {estimated_tracks.shape[2]} tracks")
    
    # Analyze results
    print("\nResults analysis:")
    num_est_tracks = estimated_tracks.shape[2]
    
    if num_est_tracks > 0:
        for i in range(num_est_tracks):
            valid_steps = ~np.isnan(estimated_tracks[0, :, i])
            track_length = np.sum(valid_steps)
            
            if track_length > 0:
                first_valid = np.where(valid_steps)[0][0]
                last_valid = np.where(valid_steps)[0][-1]
                start_pos = estimated_tracks[:2, first_valid, i]
                
                print(f"  Estimated track {i+1}:")
                print(f"    Length: {track_length} time steps")
                print(f"    First detection: step {first_valid + 1}")
                print(f"    Last detection: step {last_valid + 1}")
                print(f"    Start position: ({start_pos[0]:.1f}, {start_pos[1]:.1f})")
    else:
        print("  No tracks detected!")
    
    # Visualize results
    print("\nVisualizing results...")
    print("(Close plot window to continue)")
    
    axis_limits = [-150, 150, -150, 150]
    show_results(
        target_tracks=target_tracks,
        target_extents=target_extents,
        estimated_tracks=estimated_tracks,
        estimated_extents=estimated_extents,
        measurements=measurements,
        axis_limits=axis_limits,
        mode=0  # Final result mode
    )
    
    # Performance summary
    print("\nPerformance Summary:")
    print(f"  True targets: {num_targets}")
    print(f"  Detected tracks: {num_est_tracks}")
    
    if num_est_tracks > 0:
        print("  ✓ FILTER IS DETECTING TRACKS!")
    else:
        print("  ✗ FILTER FAILED TO DETECT TRACKS")
    
    return estimated_tracks, estimated_extents

def test_simple_scenario():
    """
    Test with a simpler scenario for debugging.
    """
    print("\n" + "=" * 60)
    print("Testing Simple Scenario")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Simple parameters
    parameters = generate_default_parameters()
    parameters = set_prior_extent(parameters, mean_target_dimension=3.0)
    parameters['surveillanceRegion'] = np.array([[-200, -200], [200, 200]])
    parameters['numParticles'] = 1000  # Fewer particles for faster testing
    parameters['numOuterIterations'] = 1
    
    # Single target scenario
    num_steps = 20
    num_targets = 1
    
    print(f"Simple scenario:")
    print(f"  Steps: {num_steps}")
    print(f"  Targets: {num_targets}")
    print(f"  Particles: {parameters['numParticles']}")
    
    # Generate simple data
    data_gen = DataGenerator(parameters)
    
    # Single target appearing for all steps
    appearance_from_to = np.array([[1, num_steps]]).T
    
    # Generate data
    start_states, start_matrices = data_gen.get_start_states(num_targets, 50, 5)
    target_tracks, target_extents = data_gen.generate_tracks_unknown(
        start_states, start_matrices, appearance_from_to, num_steps
    )
    measurements = data_gen.generate_cluttered_measurements(target_tracks, target_extents)
    
    print(f"Generated {sum(m.shape[1] for m in measurements)} total measurements")
    
    # Run filter
    print("Running filter on simple scenario...")
    eot_filter = ExtendedObjectFilterFixed(parameters)
    estimated_tracks, estimated_extents = eot_filter.run_filter(measurements)
    
    print(f"Detected {estimated_tracks.shape[2]} tracks")
    
    if estimated_tracks.shape[2] > 0:
        print("✓ Simple scenario: TRACKS DETECTED!")
    else:
        print("✗ Simple scenario: NO TRACKS DETECTED")

if __name__ == "__main__":
    print("Extended Object Tracking - Fixed Implementation Test")
    print("This tests the fixed filter with proper new target detection")
    print()
    
    # Run simple test first
    test_simple_scenario()
    
    # Then run full test
    print()
    estimated_tracks, estimated_extents = test_fixed_filter()
    
    print("\nTest completed!")