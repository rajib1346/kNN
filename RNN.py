import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances
from collections import defaultdict

def compute_density(points, distances, neighbors_count):
    return np.mean(distances[points, :neighbors_count], axis=1)

def reduced_nearest_neighbor(data, termination_ratio=0.5):
    distances = pairwise_distances(data)
    indices = np.argsort(distances, axis=1)

    remaining_points = set(range(data.shape[0]))
    while len(remaining_points) > termination_ratio * data.shape[0]:
        # For simplicity, just consider the immediate neighbor for density computation
        densities = compute_density(list(remaining_points), distances, 2)

        # Remove the point with the maximum density
        point_to_remove = np.argmax(densities)
        remaining_points.remove(point_to_remove)

    return np.array(list(remaining_points))

# Load iris data
iris = load_iris()
X = iris.data

# Apply RNN
reduced_indices = reduced_nearest_neighbor(X)
reduced_data = X[reduced_indices]

print("Original dataset size:", X.shape[0])
print("Reduced dataset size:", reduced_data.shape[0])
