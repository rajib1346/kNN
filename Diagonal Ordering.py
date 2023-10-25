import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from blist import sortedlist

class DiagonalOrdering:
    def __init__(self, n_clusters=3, n_components=2):
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.index = {}
        for i in range(n_clusters):
            self.index[i] = sortedlist()

    def fit(self, X):
        # Transform data with PCA
        X_pca = self.pca.fit_transform(X)
        
        # Cluster the data
        labels = self.kmeans.fit_predict(X_pca)
        
        # Arrange each cluster in diagonal order and store in B+-tree like structure
        for i, label in enumerate(labels):
            diagonal_value = sum(X_pca[i])
            self.index[label].add((diagonal_value, X[i]))

    def knn_search(self, query_point, k=1):
        query_transformed = self.pca.transform([query_point])[0]
        label = self.kmeans.predict([query_transformed])[0]
        diagonal_value = sum(query_transformed)

        # Find approximate position of query in diagonal order
        pos = self.index[label].bisect_left((diagonal_value,))

        # Search for nearest neighbors
        neighbors = []
        left, right = pos-1, pos
        while len(neighbors) < k:
            if left >= 0:
                neighbors.append(self.index[label][left][1])
                left -= 1
            if right < len(self.index[label]):
                neighbors.append(self.index[label][right][1])
                right += 1
        return neighbors[:k]


# Test using the Iris dataset
iris = datasets.load_iris()
X = iris.data

model = DiagonalOrdering()
model.fit(X)

query_point = [5.1, 3.5, 1.4, 0.2]
print(model.knn_search(query_point, k=3))
