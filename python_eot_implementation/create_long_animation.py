#!/usr/bin/env python3
"""
Create extended 100-frame visualization and animate into GIF.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PIL import Image
from eot_common import EOTParameters, get_start_states, generate_tracks_unknown, generate_cluttered_measurements
from eot_elliptical_shape import eot_elliptical_shape

def create_extended_animation():
    """Create 100-frame visualization and combine into animated GIF."""
    
    # Set up parameters for extended visualization
    np.random.seed(1)
    
    parameters = EOTParameters()
    parameters.scan_time = 0.2
    parameters.acceleration_deviation = 1.0
    parameters.survival_probability = 0.99
    parameters.mean_births = 0.01
    parameters.surveillance_region = np.array([[-150, 150], [-150, 150]])
    parameters.measurement_variance = 1.0
    parameters.mean_measurements = 7.0
    parameters.mean_clutter = 8.0
    parameters.prior_velocity_covariance = np.diag([10**2, 10**2])
    parameters.prior_extent_2 = 100.0
    parameters.prior_extent_1 = np.eye(2) * 3 * (parameters.prior_extent_2 - 3)
    parameters.degree_freedom_prediction = 20000.0
    parameters.num_particles = 1500  # Increased for better performance
    parameters.regularization_deviation = 0.0
    parameters.detection_threshold = 0.5
    parameters.threshold_pruning = 1e-3
    parameters.minimum_track_length = 1
    parameters.num_outer_iterations = 2
    
    # Create extended scenario with 5 targets over 100 steps
    num_targets = 5
    num_steps = 100
    start_radius = 75
    start_velocity = 10
    
    print("Generating extended ground truth scenario (100 frames)...")
    
    # Generate ground truth with staggered appearances
    start_states, start_matrices = get_start_states(num_targets, start_radius, start_velocity, parameters)
    
    # Staggered appearance times for more interesting dynamics
    appearance_from_to = np.array([
        [1, 95],   # Target 1: appears early, lasts almost full duration
        [10, 90],  # Target 2: appears later, disappears before end
        [20, 85],  # Target 3: mid-duration appearance
        [30, 100], # Target 4: appears mid-way, lasts to end
        [40, 80]   # Target 5: short duration in middle
    ]).T
    
    target_tracks, target_extents = generate_tracks_unknown(
        parameters, start_states, start_matrices, appearance_from_to, num_steps
    )
    
    measurements = generate_cluttered_measurements(target_tracks, target_extents, parameters)
    
    # Run EOT algorithm
    print("Running EOT algorithm for 100 time steps...")
    print("This may take a few minutes...")
    estimated_tracks, estimated_extents = eot_elliptical_shape(measurements, parameters)
    
    print(f"Algorithm completed. Got {estimated_tracks.shape[2]} estimated tracks.")
    
    # Create frames directory
    frames_dir = '/Users/lipingb/Desktop/Meyer/long_animation_frames'
    if os.path.exists(frames_dir):
        # Clear existing frames
        import shutil
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)
    
    print(f"Creating {num_steps} frame images...")
    
    # Create individual frame plots
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    
    for step in range(num_steps):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot measurements for this step
        if measurements[step].shape[1] > 0:
            ax.scatter(measurements[step][0, :], measurements[step][1, :],
                      c='lightgray', s=8, alpha=0.4, label='Measurements', zorder=1)
        
        # Plot true targets with extent ellipses
        for target in range(num_targets):
            if not np.isnan(target_tracks[0, step, target]):
                pos = target_tracks[:2, step, target]
                extent = target_extents[:, :, step, target]
                
                # Plot position
                ax.plot(pos[0], pos[1], 'o', color=colors[target], 
                       markersize=10, label=f'True Target {target+1}', zorder=4,
                       markeredgecolor='white', markeredgewidth=1)
                
                # Plot extent ellipse
                eigenvals, eigenvecs = np.linalg.eigh(extent)
                if np.all(eigenvals > 0):  # Valid extent matrix
                    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                    width, height = 4 * np.sqrt(eigenvals)  # 2-sigma ellipse
                    ellipse = patches.Ellipse(pos, width, height, angle=angle,
                                            facecolor=colors[target], alpha=0.12,
                                            edgecolor=colors[target], linewidth=2, zorder=2)
                    ax.add_patch(ellipse)
        
        # Plot estimated targets
        if step < estimated_tracks.shape[1] and estimated_tracks.shape[2] > 0:
            est_plotted = False
            for track in range(estimated_tracks.shape[2]):
                if not np.isnan(estimated_tracks[0, step, track]):
                    pos = estimated_tracks[:2, step, track]
                    
                    ax.plot(pos[0], pos[1], 'x', color='red', markersize=12,
                           markeredgewidth=3, label='Estimated' if not est_plotted else "",
                           zorder=5)
                    est_plotted = True
                    
                    # Plot estimated extent if available
                    if step < estimated_extents.shape[2] and track < estimated_extents.shape[3]:
                        extent = estimated_extents[:, :, step, track]
                        if not np.any(np.isnan(extent)):
                            eigenvals, eigenvecs = np.linalg.eigh(extent)
                            if np.all(eigenvals > 0):  # Valid extent matrix
                                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                                width, height = 4 * np.sqrt(eigenvals)
                                ellipse = patches.Ellipse(pos, width, height, angle=angle,
                                                        facecolor='none', edgecolor='red',
                                                        alpha=0.8, linewidth=2, linestyle='--', zorder=3)
                                ax.add_patch(ellipse)
        
        # Plot trajectory history (longer fading trail)
        trail_length = min(15, step + 1)  # Show last 15 positions
        for target in range(num_targets):
            trail_steps = range(max(0, step - trail_length + 1), step + 1)
            trail_positions = []
            for t in trail_steps:
                if not np.isnan(target_tracks[0, t, target]):
                    trail_positions.append(target_tracks[:2, t, target])
            
            if len(trail_positions) > 1:
                trail_positions = np.array(trail_positions).T
                # Create fading alpha values
                alphas = np.linspace(0.05, 0.5, len(trail_positions[0]))
                for i in range(len(trail_positions[0]) - 1):
                    ax.plot(trail_positions[0, i:i+2], trail_positions[1, i:i+2],
                           color=colors[target], alpha=alphas[i], linewidth=2, zorder=2)
        
        # Plot estimated trajectory history
        if estimated_tracks.shape[2] > 0:
            for track in range(min(5, estimated_tracks.shape[2])):  # Show all tracks
                trail_steps = range(max(0, step - trail_length + 1), step + 1)
                trail_positions = []
                for t in trail_steps:
                    if t < estimated_tracks.shape[1] and not np.isnan(estimated_tracks[0, t, track]):
                        trail_positions.append(estimated_tracks[:2, t, track])
                
                if len(trail_positions) > 1:
                    trail_positions = np.array(trail_positions).T
                    alphas = np.linspace(0.1, 0.8, len(trail_positions[0]))
                    for i in range(len(trail_positions[0]) - 1):
                        ax.plot(trail_positions[0, i:i+2], trail_positions[1, i:i+2],
                               color='red', alpha=alphas[i], linewidth=2, linestyle='--', zorder=3)
        
        # Formatting
        ax.set_xlim(-120, 120)
        ax.set_ylim(-120, 120)
        ax.set_xlabel('X Position', fontsize=14)
        ax.set_ylabel('Y Position', fontsize=14)
        ax.set_title(f'Extended Object Tracking - Time Step {step+1}/{num_steps}', 
                    fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add enhanced time step counter and progress bar
        ax.text(0.02, 0.98, f'Step: {step+1:03d}', transform=ax.transAxes, 
                fontsize=18, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                verticalalignment='top')
        
        # Add progress bar
        progress = (step + 1) / num_steps
        bar_width = 0.3
        bar_height = 0.02
        bar_x = 0.02
        bar_y = 0.02
        
        # Background bar
        ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width, bar_height, 
                                  transform=ax.transAxes, facecolor='lightgray', 
                                  alpha=0.7, zorder=10))
        # Progress bar
        ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width * progress, bar_height, 
                                  transform=ax.transAxes, facecolor='blue', 
                                  alpha=0.8, zorder=11))
        
        plt.tight_layout()
        
        # Save frame
        frame_path = os.path.join(frames_dir, f'frame_{step:03d}.png')
        plt.savefig(frame_path, dpi=80, bbox_inches='tight', facecolor='white')
        plt.close()
        
        if (step + 1) % 10 == 0:
            print(f"  Saved frames {step-8:03d} to {step:03d}")
    
    print("All frames saved. Creating animated GIFs...")
    
    # Create animated GIF
    frame_files = []
    for step in range(num_steps):
        frame_path = os.path.join(frames_dir, f'frame_{step:03d}.png')
        frame_files.append(frame_path)
    
    # Load images and create GIF
    print("Loading images...")
    images = []
    for i, frame_file in enumerate(frame_files):
        img = Image.open(frame_file)
        # Resize for smaller file size
        img = img.resize((800, 600), Image.Resampling.LANCZOS)
        images.append(img)
        if (i + 1) % 20 == 0:
            print(f"  Loaded {i+1}/{len(frame_files)} images")
    
    # Save as animated GIF - normal speed
    gif_path = '/Users/lipingb/Desktop/Meyer/eot_100_frames.gif'
    print("Creating normal speed GIF...")
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=300,  # 300ms per frame = ~3.3 FPS
        loop=0
    )
    
    print(f"100-frame GIF saved to: {gif_path}")
    
    # Create a fast version
    gif_fast_path = '/Users/lipingb/Desktop/Meyer/eot_100_frames_fast.gif'
    print("Creating fast speed GIF...")
    images[0].save(
        gif_fast_path,
        save_all=True,
        append_images=images[1:],
        duration=150,  # 150ms per frame = ~6.7 FPS
        loop=0
    )
    
    print(f"Fast 100-frame GIF saved to: {gif_fast_path}")
    
    # Create a very fast highlights version (every 4th frame)
    highlights = images[::4]  # Every 4th frame
    gif_highlights_path = '/Users/lipingb/Desktop/Meyer/eot_highlights.gif'
    print("Creating highlights GIF...")
    highlights[0].save(
        gif_highlights_path,
        save_all=True,
        append_images=highlights[1:],
        duration=250,  # 250ms per frame
        loop=0
    )
    
    print(f"Highlights GIF saved to: {gif_highlights_path}")
    
    # Print comprehensive summary
    print(f"\n" + "="*60)
    print("EXTENDED ANIMATION SUMMARY")
    print("="*60)
    
    print(f"\nSimulation Details:")
    print(f"- Total time steps: {num_steps}")
    print(f"- True targets: {num_targets}")
    print(f"- Estimated tracks: {estimated_tracks.shape[2]}")
    
    print(f"\nTrue Target Activity:")
    for target in range(num_targets):
        active_steps = np.sum(~np.isnan(target_tracks[0, :, target]))
        start_step = np.where(~np.isnan(target_tracks[0, :, target]))[0][0] + 1 if active_steps > 0 else 0
        end_step = np.where(~np.isnan(target_tracks[0, :, target]))[0][-1] + 1 if active_steps > 0 else 0
        print(f"  Target {target+1}: Steps {start_step:2d}-{end_step:2d} ({active_steps:2d} frames)")
    
    print(f"\nEstimated Track Performance:")
    if estimated_tracks.shape[2] > 0:
        for track in range(estimated_tracks.shape[2]):
            detections = np.sum(~np.isnan(estimated_tracks[0, :, track]))
            detection_rate = (detections / num_steps) * 100
            print(f"  Track {track+1}: {detections:3d}/{num_steps} detections ({detection_rate:5.1f}%)")
    
    total_measurements = sum(meas.shape[1] for meas in measurements)
    avg_measurements = total_measurements / num_steps
    print(f"\nMeasurement Statistics:")
    print(f"  Total measurements: {total_measurements}")
    print(f"  Average per frame: {avg_measurements:.1f}")
    
    print(f"\nGenerated Files:")
    print(f"  Individual frames: {frames_dir}/ ({num_steps} PNG files)")
    print(f"  Normal speed GIF: eot_100_frames.gif (~3.3 FPS)")
    print(f"  Fast speed GIF: eot_100_frames_fast.gif (~6.7 FPS)")
    print(f"  Highlights GIF: eot_highlights.gif (every 4th frame)")
    
    return gif_path, gif_fast_path, gif_highlights_path

if __name__ == "__main__":
    gif_normal, gif_fast, gif_highlights = create_extended_animation()