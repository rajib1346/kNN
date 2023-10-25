import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

class VAPlusFile:
    def __init__(self, n_clusters=3, n_bits=3):
        self.n_clusters = n_clusters
        self.n_bits = n_bits
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.nn = None
        self.representatives = None

    def fit(self, X):
        # Scalar Quantization
        max_vals = np.max(X, axis=0)
        min_vals = np.min(X, axis=0)
        intervals = (max_vals - min_vals) / (2**self.n_bits)

        self.representatives = (X - min_vals) // intervals
        self.representatives = self.representatives.astype(np.int32)
        
        # Clustering
        cluster_labels = self.kmeans.fit_predict(self.representatives)
        
        # Create Approximate Nearest Neighbors structure
        self.nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.representatives)

    def knn_search(self, query_point, k=1):
        # Convert query point to representative
        max_vals = np.max(self.representatives, axis=0)
        min_vals = np.min(self.representatives, axis=0)
        intervals = (max_vals - min_vals) / (2**self.n_bits)

        query_rep = (query_point - min_vals) // intervals
        query_rep = query_rep.astype(np.int32)

        # Use Approximate Nearest Neighbors for searching
        distances, indices = self.nn.kneighbors([query_rep], n_neighbors=k)
        return indices[0]


# Test using the Iris dataset
iris = datasets.load_iris()
X = iris.data

model = VAPlusFile()
model.fit(X)

query_point = [5.1, 3.5, 1.4, 0.2]
print(model.knn_search(query_point, k=3))
