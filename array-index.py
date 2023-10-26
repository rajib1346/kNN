import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

# Create a synthetic dataset
data, labels = make_blobs(n_samples=1000, centers=5, n_features=10, random_state=42)

# For simplicity, let's consider the data partitions to be based on labels (clusters)
partitions = [data[labels == i] for i in np.unique(labels)]

# Compute representative vectors (centroids) for each partition
centroids = [np.mean(partition, axis=0) for partition in partitions]

# Choose a reference point - for this example, we use the centroid of the entire dataset
reference_point = np.mean(data, axis=0)

# Compute the distance between the chosen reference point and each centroid
distances = [np.linalg.norm(reference_point - centroid) for centroid in centroids]

# Sort the partitions based on the computed distances
sorted_indices = np.argsort(distances)
sorted_partitions = [partitions[i] for i in sorted_indices]

# For a given query, find kNN points in the reduced search region
query_point = np.random.rand(data.shape[1])  # Random query point
nearest_partition_idx = np.argmin([np.linalg.norm(query_point - centroid) for centroid in centroids])
nearest_partition = sorted_partitions[nearest_partition_idx]

# Using NearestNeighbors from sklearn to find kNN in the nearest_partition
k = 5
knn = NearestNeighbors(n_neighbors=k).fit(nearest_partition)
distances, indices = knn.kneighbors(query_point.reshape(1, -1))

print("k Nearest Neighbors for the query point are:", indices[0])
