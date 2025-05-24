import numpy as np

def update_particles(old_kinematic, old_extent, old_existence, log_weights, parameters):
    """
    Update particle kinematics, extents, and existence based on measurement log-weights.

    Parameters:
        old_kinematic: np.ndarray of shape (state_dim, num_particles)
        old_extent: np.ndarray of shape (d, d, num_particles)
        old_existence: float
        log_weights: np.ndarray (M, num_particles) or (num_particles,) of log-likelihoods
        parameters: dict with keys ['numParticles', 'regularizationDeviation']

    Returns:
        updated_kinematic, updated_extent, updated_existence
    """
    num_particles = parameters['numParticles']
    reg_dev = parameters['regularizationDeviation']

    # Aggregate log-weights across measurements
    lw = np.sum(log_weights, axis=1) if log_weights.ndim > 1 else log_weights
    # Compute existence update
    alive_update = np.mean(np.exp(lw))
    if np.isinf(alive_update):
        updated_existence = 1.0
    else:
        alive = old_existence * alive_update
        dead = 1.0 - old_existence
        updated_existence = alive / (dead + alive)

    if updated_existence != 0.0:
        # Normalize weights
        lw = lw - np.max(lw)
        weights = np.exp(lw)
        weights /= np.sum(weights)
        # Resample particles
        indices = resample_systematic(weights, num_particles)
        updated_kinematic = old_kinematic[:, indices]
        updated_extent = old_extent[:, :, indices]
        # Add regularization noise
        updated_kinematic[0:2, :] += reg_dev * np.random.randn(2, num_particles)
    else:
        updated_kinematic = np.full_like(old_kinematic, np.nan)
        updated_extent = np.full_like(old_extent, np.nan)

    return updated_kinematic, updated_extent, updated_existence
