import numpy as np
from collections import defaultdict
from sklearn.datasets import load_iris
from sklearn.metrics import euclidean_distances
from operator import itemgetter
from functools import lru_cache

class SimpleKNN:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.cache = defaultdict(int)

    @lru_cache(maxsize=100)  # Using Python's built-in caching mechanism as an example
    def knn_search(self, point, k=3):
        distances = []
        for index, sample in enumerate(self.data):
            distance = euclidean_distances([point], [sample])
            distances.append((index, distance))
            
            # Count the frequency of data point access
            self.cache[index] += 1

        # Sort distances and get the top k indices
        sorted_distances = sorted(distances, key=itemgetter(1))[:k]
        indices = [index for index, _ in sorted_distances]

        # Return the nearest neighbors' target values
        return [self.target[index] for index in indices]
    
    def get_most_frequent_data_points(self, n=5):
        # Get the n most accessed data points from the cache
        sorted_cache = sorted(self.cache.items(), key=itemgetter(1), reverse=True)
        return [self.data[index] for index, _ in sorted_cache[:n]]

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create an instance and search for nearest neighbors
knn = SimpleKNN(X, y)
print(knn.knn_search(X[50], k=5))
print(knn.get_most_frequent_data_points(n=5))
