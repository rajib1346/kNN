import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import heapq

# Generate a random dataset
data, _ = make_blobs(n_samples=1000, centers=5, random_state=42)

# QuadTree Node definition
class Node:
    def __init__(self, data, depth=0, top_left=None, bottom_right=None):
        self.data = data
        self.children = []
        self.depth = depth
        self.top_left = top_left if top_left else [0, 0]
        self.bottom_right = bottom_right if bottom_right else [100, 100]
        
    def is_leaf(self):
        return len(self.children) == 0

# QuadTree implementation
class QuadTree:
    def __init__(self, capacity=4):
        self.root = Node(data=None)
        self.capacity = capacity
        
    def insert(self, point):
        self._insert(point, self.root)
        
    def _insert(self, point, node):
        if node.is_leaf():
            if node.data is None:
                node.data = point
            else:
                if len(node.data) < self.capacity:
                    node.data.append(point)
                else:
                    self.split(node)
                    for p in node.data:
                        self._insert(p, node)
                    self._insert(point, node)
        else:
            for child in node.children:
                if self.in_boundary(point, child):
                    self._insert(point, child)
                    break
                    
    def split(self, node):
        x_mid = (node.top_left[0] + node.bottom_right[0]) / 2
        y_mid = (node.top_left[1] + node.bottom_right[1]) / 2
        
        tl_node = Node(data=[], depth=node.depth+1,
                      top_left=node.top_left, bottom_right=[x_mid, y_mid])
        tr_node = Node(data=[], depth=node.depth+1,
                      top_left=[x_mid, node.top_left[1]], bottom_right=[node.bottom_right[0], y_mid])
        bl_node = Node(data=[], depth=node.depth+1,
                      top_left=[node.top_left[0], y_mid], bottom_right=[x_mid, node.bottom_right[1]])
        br_node = Node(data=[], depth=node.depth+1,
                      top_left=[x_mid, y_mid], bottom_right=node.bottom_right)
        
        node.children = [tl_node, tr_node, bl_node, br_node]
        node.data = []
        
    def in_boundary(self, point, node):
        return (node.top_left[0] <= point[0] < node.bottom_right[0] and
                node.top_left[1] <= point[1] < node.bottom_right[1])

# Build the tree
qt = QuadTree()
for point in data:
    qt.insert(point)

# kNN search
def knn_search(tree, query_point, k):
    closest_points = []
    
    def _search(node, query_point):
        nonlocal closest_points
        if node.is_leaf() and node.data:
            for point in node.data:
                if len(closest_points) < k:
                    heapq.heappush(closest_points, (-np.linalg.norm(np.array(query_point) - np.array(point)), point))
                else:
                    current_farthest_dist = -closest_points[0][0]
                    if np.linalg.norm(np.array(query_point) - np.array(point)) < current_farthest_dist:
                        heapq.heappop(closest_points)
                        heapq.heappush(closest_points, (-np.linalg.norm(np.array(query_point) - np.array(point)), point))
        else:
            for child in node.children:
                if tree.in_boundary(query_point, child):
                    _search(child, query_point)
                    break
                    
    _search(tree.root, query_point)
    
    return [item[1] for item in closest_points]

# Test kNN
k = 5
query = [50, 50]
result = knn_search(qt, query, k)

# Plot the results
plt.scatter(data[:, 0], data[:, 1], s=5, color='blue')
plt.scatter(query[0], query[1], s=100, color='red', marker='x')
for pt in result:
    plt.scatter(pt[0], pt[1], s=50, color='green')
plt.show()
