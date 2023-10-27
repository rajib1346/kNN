import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.metrics import pairwise_distances

# Load the dataset
data = fetch_20newsgroups_vectorized(subset='all').data.toarray()[:1000]

# Define HDR-Tree class
class HDRTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def build_tree(self, data, depth=1):
        n, d = data.shape
        if depth > self.max_depth or n <= 5:
            return {
                'data': data,
                'is_leaf': True
            }

        # Apply PCA and k-means
        pca = PCA(n_components=2)
        transformed_data = pca.fit_transform(data)
        kmeans = KMeans(n_clusters=2)
        clusters = kmeans.fit_predict(transformed_data)
        
        return {
            'left_child': self.build_tree(data[clusters == 0], depth + 1),
            'right_child': self.build_tree(data[clusters == 1], depth + 1),
            'centroid': kmeans.cluster_centers_,
            'is_leaf': False
        }

    def build(self, data):
        self.tree = self.build_tree(data)

    def search(self, point, node=None, maxdknn=np.inf):
        if node is None:
            node = self.tree

        if node['is_leaf']:
            distances = pairwise_distances([point], node['data'])[0]
            return distances.min()

        left_distance = pairwise_distances([point], [node['left_child']['centroid']])[0][0]
        right_distance = pairwise_distances([point], [node['right_child']['centroid']])[0][0]

        min_distance = np.inf
        if left_distance < maxdknn:
            min_distance = min(min_distance, self.search(point, node['left_child'], maxdknn))
        if right_distance < maxdknn:
            min_distance = min(min_distance, self.search(point, node['right_child'], maxdknn))
        
        return min_distance

tree = HDRTree()
tree.build(data)

# Test
test_point = data[101]
distance = tree.search(test_point)
print("Minimum distance to test point:", distance)
