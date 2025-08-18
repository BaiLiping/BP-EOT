"""
Common utility functions for Extended Object Tracking (EOT) algorithm.

This module contains helper functions and data structures used by the main EOT algorithm.
Each function is documented with clear explanations of the underlying mathematics.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from scipy.stats import multivariate_normal, poisson, wishart, invwishart
from scipy.linalg import sqrtm, cholesky
import matplotlib.patches as patches


@dataclass
class EOTParameters:
    """
    Data class containing all parameters for the EOT algorithm.
    
    This structure organizes the many parameters needed for the tracking algorithm
    into logical groups for better maintainability.
    """
    
    # Temporal parameters
    scan_time: float = 0.2
    
    # Target dynamics parameters  
    acceleration_deviation: float = 1.0
    survival_probability: float = 0.99
    mean_births: float = 0.01
    
    # Measurement parameters
    surveillance_region: np.ndarray = None
    measurement_variance: float = 1.0
    mean_measurements: float = 8.0
    mean_clutter: float = 10.0
    
    # Prior distribution parameters
    prior_velocity_covariance: np.ndarray = None
    prior_extent_1: np.ndarray = None
    prior_extent_2: float = 100.0
    degree_freedom_prediction: float = 20000.0
    
    # Sampling parameters
    num_particles: int = 5000
    regularization_deviation: float = 0.0
    
    # Detection and pruning parameters
    detection_threshold: float = 0.5
    threshold_pruning: float = 1e-3
    minimum_track_length: int = 1
    
    # Message passing parameters
    num_outer_iterations: int = 2


def get_transition_matrices(scan_time: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate state transition matrices for constant velocity motion model.
    
    This implements a 2D constant velocity motion model with acceleration noise.
    The state vector is [x, y, vx, vy] where (x,y) is position and (vx,vy) is velocity.
    
    Args:
        scan_time: Time interval between measurements
        
    Returns:
        A: State transition matrix (4x4)
        W: Process noise input matrix (4x2)
        
    Mathematical model:
        x(k+1) = A * x(k) + W * w(k)
        where w(k) is 2D acceleration noise
    """
    # State transition matrix for [x, y, vx, vy]
    A = np.eye(4)
    A[0, 2] = scan_time  # x = x + vx * dt
    A[1, 3] = scan_time  # y = y + vy * dt
    
    # Process noise input matrix (maps 2D acceleration to 4D state)
    W = np.zeros((4, 2))
    W[0, 0] = 0.5 * scan_time**2  # x component from x-acceleration
    W[1, 1] = 0.5 * scan_time**2  # y component from y-acceleration  
    W[2, 0] = scan_time           # vx component from x-acceleration
    W[3, 1] = scan_time           # vy component from y-acceleration
    
    return A, W


def get_start_states(num_targets: int, radius: float, speed: float, 
                    parameters: EOTParameters) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate initial target states arranged in a circle.
    
    This function creates initial conditions for multiple targets arranged
    in a circular formation, each moving tangentially to the circle.
    
    Args:
        num_targets: Number of targets to initialize
        radius: Radius of circular formation
        speed: Initial speed of targets
        parameters: EOT parameters containing prior extent information
        
    Returns:
        start_states: Initial kinematic states (4 x num_targets)
        start_matrices: Initial extent matrices (2 x 2 x num_targets)
    """
    # Generate initial extent matrices from inverse Wishart distribution
    start_matrices = np.zeros((2, 2, num_targets))
    for target in range(num_targets):
        # Sample extent matrix from inverse Wishart prior
        start_matrices[:, :, target] = invwishart.rvs(
            df=parameters.prior_extent_2,
            scale=parameters.prior_extent_1
        )
    
    # Special case for single target
    if num_targets < 2:
        start_states = np.array([[0], [radius], [0], [-speed]])
        return start_states, start_matrices
    
    # Initialize state array: [x, y, vx, vy] for each target
    start_states = np.zeros((4, num_targets))
    
    # First target starts at (0, radius) moving left
    start_states[:, 0] = [0, radius, 0, -speed]
    
    # Arrange remaining targets in circle with equal angular spacing
    step_size = 2 * np.pi / num_targets
    
    for target in range(1, num_targets):
        angle = target * step_size
        # Position on circle
        x = np.sin(angle) * radius
        y = np.cos(angle) * radius
        # Velocity tangent to circle (perpendicular to radius)
        vx = -np.sin(angle) * speed
        vy = -np.cos(angle) * speed
        
        start_states[:, target] = [x, y, vx, vy]
    
    return start_states, start_matrices


def generate_tracks_unknown(parameters: EOTParameters, start_states: np.ndarray,
                          extent_matrices: np.ndarray, appearance_from_to: np.ndarray,
                          num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ground truth target trajectories with appearance/disappearance times.
    
    This function simulates target motion using a constant velocity model with
    acceleration noise. Targets can appear and disappear at specified times.
    
    Args:
        parameters: EOT parameters
        start_states: Initial target states (4 x num_targets)
        extent_matrices: Target extent matrices (2 x 2 x num_targets)
        appearance_from_to: Appearance/disappearance times (2 x num_targets)
        num_steps: Number of simulation time steps
        
    Returns:
        target_tracks: Target trajectories (4 x num_steps x num_targets)
        target_extents: Target extent matrices (2 x 2 x num_steps x num_targets)
    """
    acceleration_deviation = parameters.acceleration_deviation
    scan_time = parameters.scan_time
    num_targets = start_states.shape[1]
    
    # Get motion model matrices
    A, W = get_transition_matrices(scan_time)
    
    # Initialize output arrays (NaN indicates target not present)
    target_tracks = np.full((4, num_steps, num_targets), np.nan)
    target_extents = np.full((2, 2, num_steps, num_targets), np.nan)
    
    # Simulate each target trajectory
    for target in range(num_targets):
        # Current state starts from initial condition
        current_state = start_states[:, target].copy()
        
        # Get appearance/disappearance times (convert to 0-indexed)
        appear_time = int(appearance_from_to[0, target] - 1)
        disappear_time = int(appearance_from_to[1, target] - 1)
        
        # Simulate trajectory for all time steps
        for step in range(num_steps):
            # Apply motion model with acceleration noise
            noise = W @ (acceleration_deviation * np.random.randn(2))
            current_state = A @ current_state + noise
            
            # Store state only if target is supposed to be visible
            if appear_time <= step <= disappear_time:
                target_tracks[:, step, target] = current_state
                target_extents[:, :, step, target] = extent_matrices[:, :, target]
    
    return target_tracks, target_extents


def generate_cluttered_measurements(target_tracks: np.ndarray, target_extents: np.ndarray,
                                  parameters: EOTParameters) -> List[np.ndarray]:
    """
    Generate noisy measurements from targets plus clutter (false alarms).
    
    This function simulates the measurement process:
    1. For each visible target, generate Poisson-distributed number of measurements
    2. Each measurement is drawn from target position + extent noise + sensor noise
    3. Add uniformly distributed clutter measurements
    4. Randomly permute all measurements
    
    Args:
        target_tracks: Ground truth target trajectories
        target_extents: Ground truth target extents  
        parameters: EOT parameters
        
    Returns:
        cluttered_measurements: List of measurement arrays, one per time step
    """
    mean_clutter = parameters.mean_clutter
    measurement_variance = parameters.measurement_variance
    mean_measurements = parameters.mean_measurements
    surveillance_region = parameters.surveillance_region
    
    num_steps = target_tracks.shape[1]
    num_targets = target_tracks.shape[2]
    
    cluttered_measurements = []
    
    for step in range(num_steps):
        measurements = np.empty((2, 0))  # Start with empty measurements
        
        # Generate measurements from each visible target
        for target in range(num_targets):
            # Skip if target not present at this time
            if np.isnan(target_tracks[0, step, target]):
                continue
                
            # Get target position and extent
            target_pos = target_tracks[:2, step, target]
            target_extent = target_extents[:, :, step, target]
            
            # Number of measurements from this target (Poisson distributed)
            num_meas = np.random.poisson(mean_measurements)
            
            if num_meas > 0:
                # Generate measurements from target extent
                # Each measurement = target_position + extent_noise + sensor_noise
                extent_noise = np.random.multivariate_normal(
                    mean=np.zeros(2),
                    cov=target_extent @ target_extent,  # extent^2 as covariance
                    size=num_meas
                ).T
                
                sensor_noise = np.sqrt(measurement_variance) * np.random.randn(2, num_meas)
                
                target_measurements = (target_pos.reshape(-1, 1) + 
                                     extent_noise + sensor_noise)
                
                # Concatenate to measurement list
                measurements = np.concatenate([measurements, target_measurements], axis=1)
        
        # Generate false alarm (clutter) measurements
        num_false_alarms = np.random.poisson(mean_clutter)
        
        if num_false_alarms > 0:
            # Uniform distribution over surveillance region
            false_alarms = np.random.uniform(
                low=[surveillance_region[0, 0], surveillance_region[1, 0]],
                high=[surveillance_region[0, 1], surveillance_region[1, 1]],
                size=(num_false_alarms, 2)
            ).T
            
            # Add false alarms to measurements
            measurements = np.concatenate([false_alarms, measurements], axis=1)
        
        # Randomly shuffle measurements (target and clutter mixed)
        if measurements.shape[1] > 0:
            perm = np.random.permutation(measurements.shape[1])
            measurements = measurements[:, perm]
        
        cluttered_measurements.append(measurements)
    
    return cluttered_measurements


def systematic_resample(weights: np.ndarray, num_particles: int) -> np.ndarray:
    """
    Perform systematic resampling of particles.
    
    Systematic resampling is a low-variance resampling method that ensures
    particles with higher weights are more likely to be selected while
    maintaining diversity in the particle set.
    
    Args:
        weights: Normalized particle weights (sum to 1)
        num_particles: Number of particles to resample
        
    Returns:
        indices: Indices of selected particles
    """
    # Cumulative sum of weights
    cumsum = np.cumsum(weights)
    
    # Generate systematic sampling points
    u = np.random.rand() / num_particles
    sampling_points = u + np.arange(num_particles) / num_particles
    
    # Find indices using searchsorted
    indices = np.searchsorted(cumsum, sampling_points)
    
    # Ensure indices are within bounds
    indices = np.minimum(indices, len(weights) - 1)
    
    return indices


def iwishart_fast_vector(scale_matrix: np.ndarray, df: float, 
                        num_samples: int) -> np.ndarray:
    """
    Fast vectorized sampling from inverse Wishart distribution.
    
    This function efficiently generates multiple samples from an inverse Wishart
    distribution, which is used for sampling target extent matrices.
    
    Args:
        scale_matrix: Scale matrix (2x2)
        df: Degrees of freedom
        num_samples: Number of samples to generate
        
    Returns:
        samples: Inverse Wishart samples (2x2xnum_samples)
    """
    # Use scipy's invwishart for simplicity and correctness
    samples = np.zeros((2, 2, num_samples))
    
    for i in range(num_samples):
        samples[:, :, i] = invwishart.rvs(df=df, scale=scale_matrix)
    
    return samples


def wishart_fast_vector(scale_matrices: np.ndarray, df: float, 
                       num_samples: int) -> np.ndarray:
    """
    Fast vectorized sampling from Wishart distribution.
    
    This generates samples from the Wishart distribution used in the
    prediction step of the extent estimation.
    
    Args:
        scale_matrices: Scale matrices (2x2xnum_samples)
        df: Degrees of freedom  
        num_samples: Number of samples
        
    Returns:
        samples: Wishart samples (2x2xnum_samples)
    """
    samples = np.zeros((2, 2, num_samples))
    
    for i in range(num_samples):
        samples[:, :, i] = wishart.rvs(df=df, scale=scale_matrices[:, :, i])
    
    return samples


def show_results(target_tracks: np.ndarray, target_extents: np.ndarray,
                estimated_tracks: np.ndarray, estimated_extents: np.ndarray,
                measurements: List[np.ndarray], plot_limits: List[float],
                mode: int = 0):
    """
    Visualize tracking results with ground truth and estimates.
    
    This function provides several visualization modes:
    - mode 0: Show final result only
    - mode 1: Automatic animation
    - mode 2: Manual frame-by-frame (press space to advance)
    
    Args:
        target_tracks: Ground truth target trajectories
        target_extents: Ground truth target extents
        estimated_tracks: Estimated target trajectories  
        estimated_extents: Estimated target extents
        measurements: List of measurements per time step
        plot_limits: Plot boundaries [x_min, x_max, y_min, y_max]
        mode: Visualization mode
    """
    
    if mode == 0:
        # Show final result only
        _plot_final_result(target_tracks, target_extents, estimated_tracks, 
                          estimated_extents, measurements, plot_limits)
    else:
        # Show frame-by-frame animation
        _plot_animation(target_tracks, target_extents, estimated_tracks,
                       estimated_extents, measurements, plot_limits, mode)


def _plot_final_result(target_tracks, target_extents, estimated_tracks,
                      estimated_extents, measurements, plot_limits):
    """Plot final tracking results."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot all measurements as gray dots
    measurement_plotted = False
    for step_measurements in measurements:
        if step_measurements.shape[1] > 0:
            ax.scatter(step_measurements[0, :], step_measurements[1, :], 
                      c='lightgray', s=1, alpha=0.3, 
                      label='Measurements' if not measurement_plotted else "")
            measurement_plotted = True
    
    # Plot true tracks
    num_targets = target_tracks.shape[2]
    colors = plt.cm.tab10(np.linspace(0, 1, num_targets))
    
    for target in range(num_targets):
        valid_steps = ~np.isnan(target_tracks[0, :, target])
        if np.any(valid_steps):
            ax.plot(target_tracks[0, valid_steps, target], 
                   target_tracks[1, valid_steps, target],
                   color=colors[target], linewidth=2, 
                   label=f'True Track {target+1}' if target < 5 else "")
    
    # Plot estimated tracks
    if estimated_tracks.shape[2] > 0:
        est_plotted = False
        for track in range(estimated_tracks.shape[2]):
            valid_steps = ~np.isnan(estimated_tracks[0, :, track])
            if np.any(valid_steps):
                ax.plot(estimated_tracks[0, valid_steps, track],
                       estimated_tracks[1, valid_steps, track],
                       '--', color='red', linewidth=3, alpha=0.8,
                       label='Estimated Tracks' if not est_plotted else "")
                est_plotted = True
    
    ax.set_xlim(plot_limits[0], plot_limits[1])
    ax.set_ylim(plot_limits[2], plot_limits[3])
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position') 
    ax.set_title('Extended Object Tracking Results')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot instead of showing it
    plt.savefig('/Users/lipingb/Desktop/Meyer/eot_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Visualization saved to: eot_results.png")


def _plot_animation(target_tracks, target_extents, estimated_tracks,
                   estimated_extents, measurements, plot_limits, mode):
    """Plot frame-by-frame animation."""
    num_steps = target_tracks.shape[1]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.ion()  # Turn on interactive mode
    
    for step in range(num_steps):
        ax.clear()
        
        # Plot measurements for this time step
        if measurements[step].shape[1] > 0:
            ax.scatter(measurements[step][0, :], measurements[step][1, :],
                      c='lightgray', s=10, alpha=0.6, label='Measurements')
        
        # Plot true target positions and extents
        num_targets = target_tracks.shape[2]
        colors = plt.cm.tab10(np.linspace(0, 1, num_targets))
        
        for target in range(num_targets):
            if not np.isnan(target_tracks[0, step, target]):
                pos = target_tracks[:2, step, target]
                extent = target_extents[:, :, step, target]
                
                # Plot position
                ax.plot(pos[0], pos[1], 'o', color=colors[target], 
                       markersize=8, label=f'True Target {target+1}' if target < 3 else "")
                
                # Plot extent ellipse
                eigenvals, eigenvecs = np.linalg.eigh(extent)
                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                width, height = 2 * np.sqrt(eigenvals)
                ellipse = patches.Ellipse(pos, width, height, angle=angle,
                                        facecolor='none', edgecolor=colors[target],
                                        alpha=0.5, linewidth=1)
                ax.add_patch(ellipse)
        
        # Plot estimated targets
        if step < estimated_tracks.shape[1] and estimated_tracks.shape[2] > 0:
            for track in range(estimated_tracks.shape[2]):
                if not np.isnan(estimated_tracks[0, step, track]):
                    pos = estimated_tracks[:2, step, track]
                    
                    ax.plot(pos[0], pos[1], 'x', color='red', markersize=10,
                           markeredgewidth=3, label='Estimated Target' if track == 0 else "")
                    
                    if step < estimated_extents.shape[2]:
                        extent = estimated_extents[:, :, step, track]
                        eigenvals, eigenvecs = np.linalg.eigh(extent)
                        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                        width, height = 2 * np.sqrt(eigenvals)
                        ellipse = patches.Ellipse(pos, width, height, angle=angle,
                                                facecolor='none', edgecolor='red',
                                                alpha=0.7, linewidth=2, linestyle='--')
                        ax.add_patch(ellipse)
        
        ax.set_xlim(plot_limits[0], plot_limits[1])
        ax.set_ylim(plot_limits[2], plot_limits[3])
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Extended Object Tracking - Time Step {step+1}/{num_steps}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.draw()
        
        if mode == 2:
            # Manual mode - wait for key press
            input("Press Enter to continue to next frame...")
        else:
            # Automatic mode
            plt.pause(0.1)
    
    plt.ioff()
    plt.show()