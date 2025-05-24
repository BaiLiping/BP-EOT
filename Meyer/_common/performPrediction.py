import numpy as np


def get_transition_matrices(scan_time):
    """
    Compute state transition matrix A and process noise matrix W for a constant velocity model.
    """
    A = np.eye(4)
    A[0, 2] = scan_time
    A[1, 3] = scan_time

    W = np.zeros((4, 2))
    W[0, 0] = 0.5 * scan_time ** 2
    W[1, 1] = 0.5 * scan_time ** 2
    W[2, 0] = scan_time
    W[3, 1] = scan_time
    return A, W


def sample_wishart(scale_matrix, df):
    """
    Sample from the Wishart distribution W_p(df, scale_matrix).
    scale_matrix: positive-definite matrix (p x p)
    df: degrees of freedom (int)
    Returns a sample (p x p) matrix.
    """
    p = scale_matrix.shape[0]
    # Cholesky decomposition of scale matrix
    L = np.linalg.cholesky(scale_matrix)
    # Generate df samples from N(0, I)
    X = np.random.randn(int(df), p)
    # Induce the desired covariance
    Y = X @ L.T
    # Wishart sample
    return Y.T @ Y


def perform_prediction(old_particles, old_existences, old_extents, scan_time, parameters):
    """
    Perform prediction step for a set of particles in a GGIW-PHD filter.
    old_particles: array of shape (4, num_particles, num_targets)
    old_existences: array of shape (num_targets,)
    old_extents: array of shape (2, 2, num_particles, num_targets)
    scan_time: time interval between scans
    parameters: dict with keys:
        - 'accelerationDeviation'
        - 'survivalProbability'
        - 'degreeFreedomPrediction'
    Returns:
    new_particles, new_existences, new_extents
    """
    num_states, num_particles, num_targets = old_particles.shape
    driving_noise_variance = parameters['accelerationDeviation'] ** 2
    survival_probability = parameters['survivalProbability']
    df_pred = parameters['degreeFreedomPrediction']

    # Pre-allocate output arrays
    new_particles = np.empty_like(old_particles)
    new_existences = np.empty_like(old_existences)
    new_extents = np.empty_like(old_extents)

    # Get motion model matrices
    A, W = get_transition_matrices(scan_time)

    # Update extents via Wishart sampling
    for t in range(num_targets):
        for i in range(num_particles):
            scale = old_extents[:, :, i, t] / df_pred
            new_extents[:, :, i, t] = sample_wishart(scale, df_pred)

    # Predict kinematic states and existence probabilities
    for t in range(num_targets):
        noise = np.sqrt(driving_noise_variance) * np.random.randn(2, num_particles)
        new_particles[:, :, t] = A @ old_particles[:, :, t] + W @ noise
        new_existences[t] = survival_probability * old_existences[t]

    return new_particles, new_existences, new_extents
