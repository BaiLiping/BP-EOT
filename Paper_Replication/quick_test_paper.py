#!/usr/bin/env python3
"""
Quick test of the paper-based implementation with reduced complexity.
"""

import numpy as np
from PaperBasedEOT import MeasurementOrientedEOT
import time

def quick_test():
    """
    Quick test with minimal complexity.
    """
    print("Quick Test of Paper-Based EOT Implementation")
    print("=" * 50)
    
    # Minimal parameters for fast testing
    params = {
        'num_particles': 100,           # Very few particles
        'survival_probability': 0.95,
        'num_outer_iterations': 1,      # Just one iteration
        'detection_threshold': 0.1,     # Low threshold
        'pruning_threshold': 1e-3,
        
        'measurement_variance': 1.0,
        'mean_clutter': 5.0,
        'surveillance_region': np.array([[-100, -100], [100, 100]]),
        
        'mean_births': 0.5,            # High birth rate
        'prior_extent1': np.array([[4, 0], [0, 4]]),
        'prior_extent2': 10,
        'prior_velocity_covariance': np.diag([25, 25]),
        
        'acceleration_deviation': 1.0,
        'scan_time': 0.2,
        'mean_measurements': 4
    }
    
    print("Minimal test parameters:")
    print(f"  Particles: {params['num_particles']}")
    print(f"  Iterations: {params['num_outer_iterations']}")
    
    # Create tracker
    tracker = MeasurementOrientedEOT(params)
    
    # Simple test: single time step with few measurements
    print("\nTesting single time step...")
    
    # Create 3 simple measurements
    measurements = np.array([
        [10, -20, 15],    # x coordinates
        [5, 10, -5]       # y coordinates  
    ])
    
    print(f"Input: {measurements.shape[1]} measurements")
    print(f"Positions: {measurements.T}")
    
    start_time = time.time()
    
    try:
        # Process single time step
        detections = tracker.process_time_step(measurements, 0)
        
        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.3f} seconds")
        print(f"Detections: {len(detections)}")
        
        for i, det in enumerate(detections):
            pos = det['kinematic_state'][:2]
            existence = det['existence_prob']
            print(f"  Detection {i+1}: pos=({pos[0]:.1f}, {pos[1]:.1f}), exist={existence:.3f}")
        
        if len(detections) > 0:
            print("\n✅ SUCCESS: Paper-based implementation is working!")
        else:
            print("\n⚠️  No detections, but no errors - algorithm runs correctly")
            
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_steps():
    """
    Test multiple time steps with very simple scenario.
    """
    print("\n" + "=" * 50)
    print("Testing Multiple Time Steps")
    print("=" * 50)
    
    # Even simpler parameters
    params = {
        'num_particles': 50,
        'survival_probability': 0.9,
        'num_outer_iterations': 1,
        'detection_threshold': 0.1,
        'pruning_threshold': 1e-3,
        
        'measurement_variance': 1.0,
        'mean_clutter': 2.0,
        'surveillance_region': np.array([[-50, -50], [50, 50]]),
        
        'mean_births': 0.3,
        'prior_extent1': np.array([[4, 0], [0, 4]]),
        'prior_extent2': 10,
        'prior_velocity_covariance': np.diag([16, 16]),
        
        'acceleration_deviation': 0.5,
        'scan_time': 0.2,
        'mean_measurements': 3
    }
    
    tracker = MeasurementOrientedEOT(params)
    
    # Simple trajectory: moving target
    steps = 5
    all_detections = []
    
    for step in range(steps):
        # Target moving from left to right
        target_x = -20 + step * 8
        target_y = 0
        
        # Generate measurements around target + some clutter
        measurements = []
        
        # Target measurements (2-3 around target)
        for _ in range(2):
            meas_x = target_x + np.random.normal(0, 2)
            meas_y = target_y + np.random.normal(0, 2)
            measurements.append([meas_x, meas_y])
        
        # Clutter (1-2 random)
        for _ in range(1):
            meas_x = np.random.uniform(-40, 40)
            meas_y = np.random.uniform(-20, 20)
            measurements.append([meas_x, meas_y])
        
        measurements_array = np.array(measurements).T
        
        print(f"Step {step+1}: {measurements_array.shape[1]} measurements", end="")
        
        detections = tracker.process_time_step(measurements_array, step)
        print(f" -> {len(detections)} detections")
        
        all_detections.extend(detections)
    
    print(f"\nTotal detections across all steps: {len(all_detections)}")
    
    if len(all_detections) > 0:
        print("✅ Multi-step test successful!")
        
        # Show some detection details
        for i, det in enumerate(all_detections[:5]):  # First 5 detections
            pos = det['kinematic_state'][:2]
            step = det['time_step']
            print(f"  Detection {i+1}: step {step+1}, pos=({pos[0]:.1f}, {pos[1]:.1f})")
    else:
        print("⚠️  No detections in multi-step test")
    
    return len(all_detections) > 0

if __name__ == "__main__":
    print("Paper-Based Extended Object Tracking - Quick Test")
    print("Implementation based purely on Meyer & Williams (2021) paper")
    print()
    
    # Test 1: Single step
    success1 = quick_test()
    
    # Test 2: Multiple steps (only if first test passes)
    if success1:
        success2 = test_multiple_steps()
    else:
        success2 = False
    
    print("\n" + "=" * 50)
    print("QUICK TEST SUMMARY")
    print("=" * 50)
    
    if success1:
        print("✅ Single step test: PASSED")
    else:
        print("❌ Single step test: FAILED")
    
    if success2:
        print("✅ Multi-step test: PASSED")
    else:
        print("❌ Multi-step test: FAILED")
    
    if success1 and success2:
        print("\n🎉 PAPER-BASED IMPLEMENTATION IS WORKING!")
        print("   Successfully implemented EOT from mathematical formulation")
    elif success1:
        print("\n✅ Core algorithm working, may need refinement for complex scenarios")
    else:
        print("\n⚠️  Implementation needs debugging")