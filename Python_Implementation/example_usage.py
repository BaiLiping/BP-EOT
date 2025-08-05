#!/usr/bin/env python3
"""
Simple example demonstrating usage of the Extended Object Tracking Python classes.
"""

import numpy as np
from DataGenerator import DataGenerator
from ExtendedObjectFilter import ExtendedObjectFilter
from utils import generate_default_parameters, set_prior_extent

def simple_example():
    """
    Simple example with minimal setup
    """
    print("Extended Object Tracking - Simple Example")
    print("-" * 40)
    
    # Basic parameters
    parameters = generate_default_parameters()
    parameters = set_prior_extent(parameters, mean_target_dimension=2.0)
    
    # Reduce complexity for quick test
    parameters['numParticles'] = 1000
    parameters['numOuterIterations'] = 1
    
    # Create single target scenario
    num_steps = 20
    num_targets = 2
    start_radius = 30
    start_velocity = 5
    
    # Target appears from step 1 to 20
    appearance_from_to = np.array([[1, 20], [5, 18]]).T
    
    print(f"Simulating {num_targets} targets for {num_steps} steps...")
    
    # Generate data
    data_gen = DataGenerator(parameters)
    target_tracks, target_extents, measurements = data_gen.simulate_scenario(
        num_steps, num_targets, 2.0, start_radius, start_velocity, appearance_from_to
    )
    
    print(f"Generated measurements for {len(measurements)} time steps")
    
    # Run filter
    print("Running filter...")
    eot_filter = ExtendedObjectFilter(parameters)
    estimated_tracks, estimated_extents = eot_filter.run_filter(measurements)
    
    print(f"Filter generated {estimated_tracks.shape[2]} tracks")
    
    # Print some results
    print("\nResults Summary:")
    for i in range(num_targets):
        true_valid = ~np.isnan(target_tracks[0, :, i])
        print(f"True target {i+1}: {np.sum(true_valid)} time steps")
        
    for i in range(estimated_tracks.shape[2]):
        est_valid = ~np.isnan(estimated_tracks[0, :, i])
        print(f"Estimated track {i+1}: {np.sum(est_valid)} time steps")
    
    print("\nExample completed!")
    
    return target_tracks, target_extents, measurements, estimated_tracks, estimated_extents

if __name__ == "__main__":
    simple_example()