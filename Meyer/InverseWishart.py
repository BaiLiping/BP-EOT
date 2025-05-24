import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Function to sample from a Wishart distribution
def sample_wishart(nu, Sigma, size):
    d = Sigma.shape[0]
    chol = np.linalg.cholesky(Sigma)
    samples = np.empty((size, d, d))
    for i in range(size):
        Z = np.random.randn(nu, d) @ chol.T
        samples[i] = Z.T @ Z
    return samples

# Parameters
nu = 5
Sigma = np.eye(2)
num_samples = 20  # number of ellipses to plot

# Draw samples
samples = sample_wishart(nu, Sigma, num_samples)

# Plot ellipses for each sample
fig, ax = plt.subplots()
for X in samples:
    vals, vecs = np.linalg.eigh(X)
    width, height = 2 * np.sqrt(vals)  # 1-sigma axes
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ellipse = Ellipse(xy=(0, 0), width=width, height=height, angle=angle, alpha=0.2)
    ax.add_patch(ellipse)

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
plt.title("Sampled Wishart Ellipses (ν=5, Σ=I)")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.show()
