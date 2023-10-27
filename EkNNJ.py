import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.metrics import pairwise_distances
from collections import defaultdict

# Load the dataset
data = fetch_20newsgroups_vectorized(subset='all').data.toarray()[:1000]

# Define EkNNJ HDR-Tree class
class EkNNJHDRTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None
        self.batch_updates = []
        self.RkNN_table = defaultdict(list)  # Reversed kNN table

    def build_tree(self, data, depth=1):
        n, d = data.shape
        if depth > self.max_depth or n <= 5:
            return {
                'data': data,
                'is_leaf': True,
                'dirty': False
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
            'is_leaf': False,
            'dirty': False
        }

    def build(self, data):
        self.tree = self.build_tree(data)

    def mark_dirty(self, node):
        if not node:
            return
        node['dirty'] = True
        self.mark_dirty(node.get('left_child'))
        self.mark_dirty(node.get('right_child'))

    def lazy_update(self, node=None):
        if not node:
            node = self.tree
        if not node['dirty']:
            return
        # Implement your logic for updating dirty nodes.
        # Clear the dirty flag after processing.
        node['dirty'] = False

    def batch_insert(self, points):
        self.batch_updates.extend(points)
        if len(self.batch_updates) > 10:  # For demonstration, we update after 10 points.
            self.update_tree_with_batch()
            self.batch_updates.clear()

    def update_tree_with_batch(self):
        # Implement your logic for batch update.
        # Update RkNN table accordingly.
        pass

    def delete(self, point):
        # Implement your deletion logic using the RkNN table.
        pass

    def search(self, point, node=None, maxdknn=np.inf):
        if node is None:
            node = self.tree

        if node['dirty']:
            self.lazy_update(node)

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

tree = EkNNJHDRTree()
tree.build(data)

# Test
test_point = data[101]
distance = tree.search(test_point)
print("Minimum distance to test point:", distance)
