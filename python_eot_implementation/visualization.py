"""
Visualization utilities for Extended Object Tracking results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List


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
    plt.savefig('/Users/lipingb/Desktop/BP-EOT/eot_results.png', dpi=150, bbox_inches='tight')
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