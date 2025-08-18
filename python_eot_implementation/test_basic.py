#!/usr/bin/env python3
"""
Basic test of the EOT implementation with minimal parameters.
"""

import numpy as np
from eot_common import EOTParameters, get_start_states
from eot_elliptical_shape import eot_elliptical_shape

def test_basic():
    """Test basic functionality with minimal setup."""
    
    np.random.seed(1)
    
    # Simple parameters
    parameters = EOTParameters()
    parameters.scan_time = 0.2
    parameters.acceleration_deviation = 1.0
    parameters.survival_probability = 0.99
    parameters.mean_births = 0.01
    parameters.surveillance_region = np.array([[-100, 100], [-100, 100]])
    parameters.measurement_variance = 1.0
    parameters.mean_measurements = 5.0
    parameters.mean_clutter = 5.0
    parameters.prior_velocity_covariance = np.diag([10**2, 10**2])
    parameters.prior_extent_2 = 100.0
    parameters.prior_extent_1 = np.eye(2) * 3 * (parameters.prior_extent_2 - 3)
    parameters.degree_freedom_prediction = 20000.0
    parameters.num_particles = 100  # Reduced for testing
    parameters.regularization_deviation = 0.0
    parameters.detection_threshold = 0.5
    parameters.threshold_pruning = 1e-3
    parameters.minimum_track_length = 1
    parameters.num_outer_iterations = 2
    
    # Create simple test measurements
    num_steps = 5
    measurements_cell = []
    
    for step in range(num_steps):
        # Create a few measurements near the origin
        if step < 3:
            meas = np.array([[0, 5], [0, 5]]) + 2 * np.random.randn(2, 2)
            measurements_cell.append(meas)
        else:
            # No measurements in later steps
            measurements_cell.append(np.empty((2, 0)))
    
    print("Running basic EOT test...")
    try:
        estimated_tracks, estimated_extents = eot_elliptical_shape(measurements_cell, parameters)
        print(f"Test successful! Got {estimated_tracks.shape[2]} estimated tracks")
        print(f"Track shape: {estimated_tracks.shape}")
        print(f"Extent shape: {estimated_extents.shape}")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic()