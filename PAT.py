import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

class PATNode:
    def __init__(self, points, axis=None, threshold=None, left=None, right=None):
        self.points = points
        self.axis = axis
        self.threshold = threshold
        self.left = left
        self.right = right

def build_pat_tree(points, min_points=5):
    if len(points) <= min_points:
        return PATNode(points)

    pca = PCA(n_components=1)
    pca.fit(points)
    projection = pca.transform(points).flatten()

    median = np.median(projection)
    left_points = points[projection <= median]
    right_points = points[projection > median]

    return PATNode(
        points,
        pca.components_[0],
        median,
        build_pat_tree(left_points),
        build_pat_tree(right_points)
    )

def search_pat_tree(node, query_point, k=1):
    if node.left is None and node.right is None:
        distances = np.linalg.norm(node.points - query_point, axis=1)
        indices = np.argsort(distances)[:k]
        return node.points[indices]

    proj_val = np.dot(query_point, node.axis)
    if proj_val <= node.threshold:
        first, second = node.left, node.right
    else:
        first, second = node.right, node.left

    nearest = search_pat_tree(first, query_point, k)

    # Elimination criterion
    if len(nearest) < k or np.abs(proj_val - node.threshold) < np.linalg.norm(nearest[-1] - query_point):
        nearest_second = search_pat_tree(second, query_point, k)
        nearest = np.vstack((nearest, nearest_second))
        distances = np.linalg.norm(nearest - query_point, axis=1)
        indices = np.argsort(distances)[:k]
        nearest = nearest[indices]

    return nearest

# Load the Iris dataset
data = load_iris().data
root = build_pat_tree(data)

# Query a test point
test_point = np.array([5.1, 3.5, 1.4, 0.2])
nearest_points = search_pat_tree(root, test_point, k=3)
print(f"The 3 nearest points to the test point are:\n{nearest_points}")
