import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# Define the dimensions of the grid
grid_size = 50
x = np.linspace(0, 1, grid_size)
y = np.linspace(0, 1, grid_size)
z = np.linspace(0, 1, grid_size)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Function to add a Gaussian splat
def add_gaussian_splat(accum_grid, center, amplitude, sigma):
    """ Adds a Gaussian splat to the 3D grid. """
    cx, cy, cz = center
    gaussian = amplitude * multivariate_normal.pdf(np.stack((X, Y, Z), axis=-1), mean=[cx, cy, cz], cov=sigma)
    accum_grid += gaussian

# Initialize the grid
volume = np.zeros((grid_size, grid_size, grid_size))

# Parameters for Gaussian splats
centers = [(0.5, 0.5, 0.5), (0.3, 0.3, 0.3), (0.7, 0.7, 0.7)]
amplitudes = [1, 0.5, 0.8]
sigmas = [0.01, 0.02, 0.015]  # Smaller values mean narrower splats

# Add splats to the grid
for center, amplitude, sigma in zip(centers, amplitudes, sigmas):
    add_gaussian_splat(volume, center, amplitude, sigma)

# Plotting the result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot only points with values above a threshold to avoid clutter
threshold = 0.1
mask = volume > threshold
ax.scatter(X[mask], Y[mask], Z[mask], c=volume[mask], cmap='viridis')
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_zlabel('Z coordinate')
plt.title('3D Gaussian Splatting Visualization')
plt.show()
