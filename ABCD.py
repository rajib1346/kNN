import numpy as np
from sklearn.datasets import make_blobs
from scipy.spatial import Voronoi
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

# Create a synthetic dataset
data, _ = make_blobs(n_samples=1000, centers=5, n_features=10, random_state=42)

# Generate Voronoi clusters
vor = Voronoi(data)
clusters = [data[vor.point_region == i] for i in range(len(vor.point_region))]

# Compute the distances between each cluster and its hyperplanes (ridges)
cluster_hyperplane_distances = {}
for i, ridge_points in enumerate(vor.ridge_points):
    cluster_idx = vor.point_region[ridge_points[0]]
    distance = np.linalg.norm(data[ridge_points[0]] - data[ridge_points[1]])
    cluster_hyperplane_distances[cluster_idx] = min(distance, cluster_hyperplane_distances.get(cluster_idx, float('inf')))

# kNN Search using ACDB approach
def ACDB_kNN(query, k=5):
    # Dynamically compute lower distance bounds for each cluster
    lower_bounds = [np.linalg.norm(np.mean(cluster, axis=0) - query) - cluster_hyperplane_distances[i] for i, cluster in enumerate(clusters)]
    sorted_cluster_indices = np.argsort(lower_bounds)
    
    knn_result = None
    for cluster_idx in sorted_cluster_indices:
        cluster = clusters[cluster_idx]
        knn = NearestNeighbors(n_neighbors=k).fit(cluster)
        distances, indices = knn.kneighbors(query.reshape(1, -1))
        if knn_result is None:
            knn_result = (distances, indices)
        else:
            # Merge the results using triangle inequality
            combined_distances = np.hstack((knn_result[0], distances))
            combined_indices = np.hstack((knn_result[1], indices))
            sorted_idx = np.argsort(combined_distances)
            knn_result = (combined_distances[sorted_idx][:,:k], combined_indices[sorted_idx][:,:k])
        
        # Termination condition
        if knn_result[0][0][-1] < lower_bounds[cluster_idx + 1]:
            break

    return knn_result[1][0]

query_point = np.random.rand(data.shape[1])  # Random query point
neighbors = ACDB_kNN(query_point, k=5)
print(neighbors)
