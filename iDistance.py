from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import euclidean_distances
import numpy as np
import bisect

# Load the Iris dataset
data = load_iris()
X = data.data

# Sample partitioning strategy (simple equally sized partitioning for simplicity)
def partition_data(X, m):
    chunk_size = len(X) // m
    partitions = [X[i:i+chunk_size] for i in range(0, len(X), chunk_size)]
    return partitions

# Reference point selection strategy (selecting the centroid for simplicity)
def select_reference_point(partition):
    return np.mean(partition, axis=0)

# Transform points to 1D using the reference point
def transform_to_1D(partition, reference_point):
    distances = euclidean_distances(partition, [reference_point])
    return distances.ravel()

m = 3  # Number of partitions
partitions = partition_data(X, m)

reference_points = [select_reference_point(part) for part in partitions]

# Transform all partitions to 1D
one_d_spaces = [transform_to_1D(partitions[i], reference_points[i]) for i in range(m)]

# B+-tree can be simulated using sorted lists for simplicity
sorted_spaces = [sorted(space) for space in one_d_spaces]

# Searching in 1D space
def iDistance_search(query_point, k):
    distances_to_reference = euclidean_distances([query_point], reference_points)
    sorted_indices = np.argsort(distances_to_reference).ravel()
    
    kNN = []
    
    for index in sorted_indices:
        one_d_value = euclidean_distances([query_point], [reference_points[index]])[0][0]
        position = bisect.bisect_left(sorted_spaces[index], one_d_value)
        
        # Check points around the found position
        left, right = position - 1, position + 1
        while len(kNN) < k:
            if left >= 0:
                kNN.append((sorted_spaces[index][left], index))
                left -= 1
            if right < len(sorted_spaces[index]):
                kNN.append((sorted_spaces[index][right], index))
                right += 1
            if left < 0 and right >= len(sorted_spaces[index]):
                break
                
    kNN = sorted(kNN, key=lambda x: x[0])[:k]
    return kNN

# Test
query = X[0]
print(iDistance_search(query, 5))
