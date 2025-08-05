import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

def error_ellipse(position: np.ndarray, covariance: np.ndarray, 
                 chi2_val: float = 2.4477) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate error ellipse points for visualization.
    
    Args:
        position: Center position (2,)
        covariance: Covariance matrix (2, 2)
        chi2_val: Chi-square value for confidence level
        
    Returns:
        x_ellipse: X coordinates of ellipse
        y_ellipse: Y coordinates of ellipse
    """
    eigenvals, eigenvecs = np.linalg.eigh(covariance)
    
    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Compute ellipse parameters
    a = np.sqrt(chi2_val * eigenvals[0])
    b = np.sqrt(chi2_val * eigenvals[1])
    
    # Generate ellipse points
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = a * np.cos(theta)
    ellipse_y = b * np.sin(theta)
    
    # Rotate ellipse
    ellipse_points = eigenvecs @ np.vstack([ellipse_x, ellipse_y])
    
    # Translate to position
    x_ellipse = ellipse_points[0, :] + position[0]
    y_ellipse = ellipse_points[1, :] + position[1]
    
    return x_ellipse, y_ellipse

def show_results(target_tracks: np.ndarray, target_extents: np.ndarray,
                estimated_tracks: np.ndarray, estimated_extents: np.ndarray,
                measurements: List[np.ndarray], axis_limits: List[float] = None,
                mode: int = 0) -> None:
    """
    Visualize tracking results.
    Port of showResults.m
    
    Args:
        target_tracks: True target tracks (4, num_steps, num_targets)
        target_extents: True target extents (2, 2, num_steps, num_targets)
        estimated_tracks: Estimated tracks (4, num_steps, num_tracks)
        estimated_extents: Estimated extents (2, 2, num_steps, num_tracks)
        measurements: List of measurements for each time step
        axis_limits: [x_min, x_max, y_min, y_max]
        mode: Visualization mode (0=final, 1=animated, 2=step-by-step)
    """
    num_steps = len(measurements)
    num_true_targets = target_tracks.shape[2] if target_tracks.ndim == 3 else 0
    num_est_tracks = estimated_tracks.shape[2] if estimated_tracks.ndim == 3 else 0
    
    plt.figure(figsize=(12, 8))
    
    if mode == 0:  # Final result
        plt.clf()
        
        # Plot true tracks
        for target in range(num_true_targets):
            valid_steps = ~np.isnan(target_tracks[0, :, target])
            plt.plot(target_tracks[0, valid_steps, target], 
                    target_tracks[1, valid_steps, target], 
                    'r-', linewidth=2, label=f'True Track {target+1}' if target == 0 else "")
        
        # Plot estimated tracks
        for track in range(num_est_tracks):
            valid_steps = ~np.isnan(estimated_tracks[0, :, track])
            plt.plot(estimated_tracks[0, valid_steps, track], 
                    estimated_tracks[1, valid_steps, track], 
                    'b--', linewidth=2, label=f'Est Track {track+1}' if track == 0 else "")
        
        # Plot all measurements
        for step in range(num_steps):
            if measurements[step].shape[1] > 0:
                plt.scatter(measurements[step][0, :], measurements[step][1, :], 
                          c='k', s=10, alpha=0.3)
        
        if axis_limits:
            plt.xlim(axis_limits[0], axis_limits[1])
            plt.ylim(axis_limits[2], axis_limits[3])
        
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Extended Object Tracking Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()
        
    elif mode == 1:  # Animated
        plt.ion()
        for step in range(num_steps):
            plt.clf()
            
            # Plot true tracks up to current step
            for target in range(num_true_targets):
                valid_steps = (~np.isnan(target_tracks[0, :step+1, target]) & 
                             (np.arange(step+1) <= step))
                if np.any(valid_steps):
                    plt.plot(target_tracks[0, valid_steps, target], 
                            target_tracks[1, valid_steps, target], 
                            'r-', linewidth=2, label=f'True {target+1}' if target == 0 else "")
                    
                    # Current position
                    if not np.isnan(target_tracks[0, step, target]):
                        plt.plot(target_tracks[0, step, target], 
                                target_tracks[1, step, target], 'ro', markersize=8)
            
            # Plot estimated tracks up to current step
            for track in range(num_est_tracks):
                valid_steps = (~np.isnan(estimated_tracks[0, :step+1, track]) & 
                             (np.arange(step+1) <= step))
                if np.any(valid_steps):
                    plt.plot(estimated_tracks[0, valid_steps, track], 
                            estimated_tracks[1, valid_steps, track], 
                            'b--', linewidth=2, label=f'Est {track+1}' if track == 0 else "")
                    
                    # Current position
                    if not np.isnan(estimated_tracks[0, step, track]):
                        plt.plot(estimated_tracks[0, step, track], 
                                estimated_tracks[1, step, track], 'bs', markersize=8)
            
            # Plot current measurements
            if measurements[step].shape[1] > 0:
                plt.scatter(measurements[step][0, :], measurements[step][1, :], 
                          c='k', s=20, alpha=0.7, label='Measurements' if step == 0 else "")
            
            if axis_limits:
                plt.xlim(axis_limits[0], axis_limits[1])
                plt.ylim(axis_limits[2], axis_limits[3])
            
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title(f'Step {step+1}/{num_steps}')
            if step == 0:
                plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.pause(0.1)
        
        plt.ioff()
        plt.show()

def mvn_pdf_log(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """
    Compute log probability density of multivariate normal distribution.
    
    Args:
        x: Input vector (n,)
        mean: Mean vector (n,)
        cov: Covariance matrix (n, n)
        
    Returns:
        log_prob: Log probability density
    """
    n = len(x)
    diff = x - mean
    
    # Compute log determinant and inverse
    sign, log_det = np.linalg.slogdet(cov)
    if sign <= 0:
        return -np.inf
        
    try:
        cov_inv = np.linalg.inv(cov)
        quadratic = diff.T @ cov_inv @ diff
    except np.linalg.LinAlgError:
        return -np.inf
    
    log_prob = -0.5 * (n * np.log(2 * np.pi) + log_det + quadratic)
    
    return log_prob

def compute_ospa_distance(X: np.ndarray, Y: np.ndarray, 
                         c: float = 10.0, p: int = 1) -> float:
    """
    Compute Optimal Subpattern Assignment (OSPA) distance between two sets.
    
    Args:
        X: First set of points (d, n)
        Y: Second set of points (d, m)  
        c: Cut-off parameter
        p: Order parameter
        
    Returns:
        ospa_dist: OSPA distance
    """
    if X.shape[1] == 0 and Y.shape[1] == 0:
        return 0.0
    
    if X.shape[1] == 0:
        return c * (Y.shape[1] ** (1/p))
    
    if Y.shape[1] == 0:
        return c * (X.shape[1] ** (1/p))
    
    n, m = X.shape[1], Y.shape[1]
    
    # Compute distance matrix
    D = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            D[i, j] = min(c, np.linalg.norm(X[:, i] - Y[:, j]) ** p)
    
    # Hungarian algorithm would be optimal, but use greedy assignment for simplicity
    if n <= m:
        # More Y points than X points
        min_indices = np.argmin(D, axis=1)
        used = set()
        total_cost = 0
        
        for i in range(n):
            j = min_indices[i]
            while j in used:
                D[i, j] = np.inf
                j = np.argmin(D[i, :])
            used.add(j)
            total_cost += D[i, j]
        
        # Add penalty for unassigned Y points
        total_cost += (m - n) * (c ** p)
        ospa_dist = (total_cost / m) ** (1/p)
        
    else:
        # More X points than Y points
        min_indices = np.argmin(D, axis=0)
        used = set()
        total_cost = 0
        
        for j in range(m):
            i = min_indices[j]
            while i in used:
                D[i, j] = np.inf
                i = np.argmin(D[:, j])
            used.add(i)
            total_cost += D[i, j]
        
        # Add penalty for unassigned X points
        total_cost += (n - m) * (c ** p)
        ospa_dist = (total_cost / n) ** (1/p)
    
    return ospa_dist

def generate_default_parameters() -> Dict[str, Any]:
    """
    Generate default parameters matching main.m
    
    Returns:
        parameters: Dictionary of default parameters
    """
    parameters = {
        # Simulation parameters
        'scanTime': 0.2,
        'accelerationDeviation': 1.0,
        'survivalProbability': 0.99,
        'meanBirths': 0.01,
        'surveillanceRegion': np.array([[-200, -200], [200, 200]]),
        'measurementVariance': 1.0,
        'meanMeasurements': 8,
        'meanClutter': 10,
        
        # Prior distribution parameters  
        'priorVelocityCovariance': np.diag([100, 100]),  # 10^2 each
        'priorExtent2': 100,
        'priorExtent1': None,  # Will be set based on meanTargetDimension
        'degreeFreedomPrediction': 20000,
        
        # Sampling parameters
        'numParticles': 5000,
        'regularizationDeviation': 0.0,
        
        # Detection and pruning parameters
        'detectionThreshold': 0.5,
        'thresholdPruning': 1e-3,
        'minimumTrackLength': 1,
        
        # Message passing parameters
        'numOuterIterations': 2
    }
    
    return parameters

def set_prior_extent(parameters: Dict[str, Any], mean_target_dimension: float) -> Dict[str, Any]:
    """
    Set prior extent parameters based on mean target dimension.
    
    Args:
        parameters: Parameter dictionary
        mean_target_dimension: Mean target dimension
        
    Returns:
        parameters: Updated parameter dictionary
    """
    parameters = parameters.copy()
    extent_matrix = np.array([[mean_target_dimension, 0], 
                             [0, mean_target_dimension]])
    parameters['priorExtent1'] = extent_matrix * (parameters['priorExtent2'] - 3)
    
    return parameters