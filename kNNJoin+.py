# First, let's install the necessary libraries
!pip install scikit-learn numpy

from sklearn.datasets import fetch_20newsgroups_vectorized
import numpy as np

# Load the dataset
data = fetch_20newsgroups_vectorized(subset='all')

# For simplicity, take a subset
dataset = data.data[:1000].toarray()

# iDistance with Pyramid technique
class iDistance:
    def __init__(self, m):
        self.m = m  # number of partitions
        self.data = []
        self.reference_points = []

    def transform(self, dataset):
        # Dividing the data space into m parts and assigning reference points
        self.data = np.array_split(dataset, self.m)
        for partition in self.data:
            self.reference_points.append(np.mean(partition, axis=0))

        # Converting multi-dimensional data to one-dimensional
        one_dimensional_data = []
        for i, partition in enumerate(self.data):
            for point in partition:
                distance = np.linalg.norm(point - self.reference_points[i])
                one_dimensional_data.append(distance)

        # Mapping to a B+-tree (for simplicity, we'll use a sorted list)
        self.btree = sorted(one_dimensional_data)

    def find_knn(self, point):
        # Convert point to its one-dimensional value
        distances = [np.linalg.norm(point - ref) for ref in self.reference_points]
        one_dim_value = min(distances)

        # Find the closest point in the B+-tree
        idx = np.searchsorted(self.btree, one_dim_value)

        # Find the original point
        knn_idx = idx // self.m
        knn_point = self.data[knn_idx][idx % len(self.data[knn_idx])]
        
        return knn_point

# Create iDistance instance and transform the data
idistance = iDistance(10)
idistance.transform(dataset)

# Test
test_point = data.data[1001].toarray()[0]
result = idistance.find_knn(test_point)
print("Found kNN:", result)
