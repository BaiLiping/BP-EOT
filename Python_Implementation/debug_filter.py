#!/usr/bin/env python3
"""
Debug script to understand why tracks are not being detected.
"""

import numpy as np
from DataGenerator import DataGenerator
from ExtendedObjectFilterFixed import ExtendedObjectFilterFixed
from utils import generate_default_parameters, set_prior_extent

def debug_single_step():
    """
    Debug a single time step to understand the detection issue.
    """
    print("Debugging Extended Object Filter - Single Step Analysis")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Simple parameters
    parameters = generate_default_parameters()
    parameters = set_prior_extent(parameters, mean_target_dimension=3.0)
    parameters['surveillanceRegion'] = np.array([[-200, -200], [200, 200]])
    parameters['numParticles'] = 100  # Very few particles for debugging
    parameters['numOuterIterations'] = 1
    parameters['detectionThreshold'] = 0.1  # Lower threshold for debugging
    
    print("Parameters:")
    print(f"  Detection threshold: {parameters['detectionThreshold']}")
    print(f"  Pruning threshold: {parameters['thresholdPruning']}")
    print(f"  Mean births: {parameters['meanBirths']}")
    print(f"  Mean measurements: {parameters['meanMeasurements']}")
    print(f"  Mean clutter: {parameters['meanClutter']}")
    
    # Create filter
    eot_filter = ExtendedObjectFilterFixed(parameters)
    
    # Create simple measurements - just a few points clustered around origin
    measurements = [
        np.array([[0, 5, -5, 0], [0, 0, 0, 5]]),  # 4 measurements in a cross pattern
    ]
    
    print(f"\nTest measurements:")
    print(f"  Step 1: {measurements[0].shape[1]} measurements")
    print(f"  Positions: \n{measurements[0]}")
    
    # Run filter for one step
    print("\nRunning filter...")
    
    # Manually run the filtering steps to see what happens
    # Initialize variables
    current_labels = np.zeros((2, 0))
    current_particles_kinematic = np.zeros((4, parameters['numParticles'], 0))
    current_existences = np.zeros(0)
    current_particles_extent = np.zeros((2, 2, parameters['numParticles'], 0))
    
    # Process first step
    step_measurements = measurements[0]
    num_measurements = step_measurements.shape[1]
    
    print(f"\nStep 1 processing:")
    print(f"  Number of measurements: {num_measurements}")
    print(f"  Number of existing targets: {current_particles_kinematic.shape[2]}")
    
    # Get promising new targets
    new_indexes, reordered_meas = eot_filter.get_promising_new_targets(
        current_particles_kinematic, current_particles_extent,
        current_existences, step_measurements, parameters
    )
    
    print(f"  New target indices: {new_indexes}")
    print(f"  Number of new targets to initialize: {len(new_indexes)}")
    
    # Check clustering parameters
    free_th = 0.85
    cluster_th = 0.9
    min_cluster_elems = 2
    
    print(f"\nClustering parameters:")
    print(f"  Free threshold: {free_th}")
    print(f"  Cluster threshold: {cluster_th}")
    print(f"  Min cluster elements: {min_cluster_elems}")
    
    # Since no existing targets, all measurements should have high probability of being new
    probabilities_new = np.ones(num_measurements)
    print(f"\nNew target probabilities: {probabilities_new}")
    
    # Check clustering
    free_mask = probabilities_new >= free_th
    free_measurements = step_measurements[:, free_mask]
    print(f"Free measurements: {np.sum(free_mask)} out of {num_measurements}")
    
    # Check birth covariance
    mean_extent_birth = np.trace(parameters['priorExtent1'] / (parameters['priorExtent2'] - 3)) / 2
    birth_cov = parameters['measurementVariance'] * np.eye(2) + mean_extent_birth * np.eye(2)
    print(f"\nBirth covariance:")
    print(birth_cov)
    
    # Try clustering
    clusters = eot_filter.get_clusters(free_measurements, birth_cov, cluster_th)
    print(f"\nNumber of clusters found: {len(clusters)}")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i+1}: {len(cluster)} measurements - indices {cluster}")

def debug_update_step():
    """
    Debug the update step to see why existence probabilities stay low.
    """
    print("\n" + "=" * 60)
    print("Debugging Update Step")
    print("=" * 60)
    
    np.random.seed(42)
    
    parameters = generate_default_parameters()
    parameters = set_prior_extent(parameters, mean_target_dimension=3.0)
    parameters['surveillanceRegion'] = np.array([[-200, -200], [200, 200]])
    parameters['numParticles'] = 100
    
    eot_filter = ExtendedObjectFilterFixed(parameters)
    
    # Test update_particles function
    print("Testing particle update...")
    
    # Create dummy particles
    num_particles = 100
    old_particles_kinematic = np.zeros((4, num_particles))
    old_particles_kinematic[0, :] = 0  # x position
    old_particles_kinematic[1, :] = 0  # y position
    
    old_particles_extent = np.zeros((2, 2, num_particles))
    for i in range(num_particles):
        old_particles_extent[:, :, i] = np.array([[9, 0], [0, 9]])  # 3x3 extent
    
    old_existence = 0.01  # Initial existence from mean births
    
    # Create log weights - simulate good measurements
    num_measurements = 5
    log_weights = np.zeros((num_particles, num_measurements))
    
    # Make some measurements have high likelihood
    log_weights[:, 0] = -0.5  # Good measurement
    log_weights[:, 1] = -1.0  # Okay measurement
    log_weights[:, 2:] = -10.0  # Bad measurements
    
    print(f"Initial existence probability: {old_existence}")
    print(f"Log weights shape: {log_weights.shape}")
    print(f"Mean log weight for first measurement: {np.mean(log_weights[:, 0])}")
    
    # Update particles
    _, _, new_existence = eot_filter.update_particles(
        old_particles_kinematic, old_particles_extent, old_existence, log_weights
    )
    
    print(f"Updated existence probability: {new_existence}")
    
    # Check the update calculation
    log_weights_sum = np.sum(log_weights, axis=1)
    alive_update = np.mean(np.exp(log_weights_sum))
    print(f"\nAlive update factor: {alive_update}")
    
    alive = old_existence * alive_update
    dead = 1 - old_existence
    computed_existence = alive / (dead + alive)
    print(f"Manually computed existence: {computed_existence}")

def run_all_debug():
    """
    Run all debug functions.
    """
    debug_single_step()
    debug_update_step()
    
    print("\n" + "=" * 60)
    print("Debug Analysis Complete")
    print("=" * 60)
    
    print("\nKey findings:")
    print("1. Check if measurements are being clustered properly")
    print("2. Check if new targets are being initialized")
    print("3. Check if existence probabilities are being updated correctly")
    print("4. Check if detection threshold is appropriate")

if __name__ == "__main__":
    run_all_debug()