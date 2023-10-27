import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.datasets import make_blobs

# Generate random 2D data
R, _ = make_blobs(n_samples=100, centers=3, random_state=42)
S, _ = make_blobs(n_samples=100, centers=3, random_state=0)

# For simplicity, let's consider the centers of R as the pivots
pivots = np.array([R[:, 0].mean(), R[:, 1].mean()]).reshape(1, -1)

# Create Voronoi diagram
vor = Voronoi(pivots)

# Plot
plt.figure(figsize=(10, 5))

# Plotting R and S datasets
plt.scatter(R[:, 0], R[:, 1], c='red', label='R Dataset')
plt.scatter(S[:, 0], S[:, 1], c='blue', s=50, edgecolors='k', label='S Dataset')

# Plot Voronoi
voronoi_plot_2d(vor, show_vertices=False, show_points=False, ax=plt.gca())

plt.legend()
plt.title('PGBJ Partitioning using Voronoi Diagram')
plt.show()
