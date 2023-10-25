import numpy as np
from sklearn import datasets
from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA

# For the sake of this example, let's define a simple Bregman divergence based on squared Euclidean distance.
def bregman_divergence(x, y):
    return np.sum((x - y) ** 2)

class BregmanPartitionKNN:
    def __init__(self, n_partitions=3, tree_leaf_size=40):
        self.n_partitions = n_partitions
        self.tree_leaf_size = tree_leaf_size
        self.trees = []

    def fit(self, X):
        # 1. Use PCA to partition the HD space
        pca = PCA(n_components=self.n_partitions)
        X_transformed = pca.fit_transform(X)

        # Split the dataset based on the transformed space
        partitions = np.array_split(X_transformed, self.n_partitions)

        # 2. For each partition, create a BB-tree (BallTree using Bregman divergence)
        for partition in partitions:
            tree = BallTree(partition, leaf_size=self.tree_leaf_size, metric=bregman_divergence)
            self.trees.append(tree)

    def knn_search(self, query_point, k=1):
        neighbors_indices = []
        neighbors_distances = []

        for tree in self.trees:
            distances, indices = tree.query([query_point], k)
            neighbors_indices.extend(indices[0])
            neighbors_distances.extend(distances[0])

        # Sort results and get top-k results
        sorted_indices = np.argsort(neighbors_distances)[:k]
        return np.array(neighbors_indices)[sorted_indices]

# Test using the Iris dataset
iris = datasets.load_iris()
X = iris.data

model = BregmanPartitionKNN()
model.fit(X)

query_point = [5.1, 3.5, 1.4, 0.2]
print(model.knn_search(query_point, k=3))
