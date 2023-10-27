import numpy as np
from sklearn.datasets import load_iris
from queue import PriorityQueue

class BallTreeNode:
    def __init__(self, data, depth=0):
        self.data = data
        self.radius = 0.0
        self.center = np.mean(data, axis=0)
        self.left = None
        self.right = None

        if len(data) > 1:
            # Split the data into two parts
            axis = depth % data.shape[1]
            sorted_indices = np.argsort(data[:, axis])
            data = data[sorted_indices]
            midpoint = len(data) // 2

            self.left = BallTreeNode(data[:midpoint], depth + 1)
            self.right = BallTreeNode(data[midpoint:], depth + 1)

            self.radius = max(np.linalg.norm(self.left.center - self.center),
                              np.linalg.norm(self.right.center - self.center))

def ball_tree_nearest_neighbor(ball_tree, test_point, k=1):
    def knn_search(node, query_point):
        if node is None:
            return []
        
        dist_to_center = np.linalg.norm(query_point - node.center)
        
        # Check distance condition (1.)
        if dist_to_center - node.radius > result_queue.queue[0][0]:
            return []

        # If leaf node (2.)
        if node.left is None and node.right is None:
            distances = np.linalg.norm(node.data - query_point, axis=1)
            for point, distance in zip(node.data, distances):
                if distance < result_queue.queue[0][0]:
                    result_queue.get()
                    result_queue.put((-distance, point))

        # If internal node (3.)
        else:
            if np.linalg.norm(query_point - node.left.center) < np.linalg.norm(query_point - node.right.center):
                knn_search(node.left, query_point)
                knn_search(node.right, query_point)
            else:
                knn_search(node.right, query_point)
                knn_search(node.left, query_point)
        
        return [item[1] for item in result_queue.queue]

    result_queue = PriorityQueue()
    for _ in range(k):
        result_queue.put((float("inf"), None))
    neighbors = knn_search(ball_tree, test_point)
    return neighbors

# Load iris dataset for demonstration
data = load_iris().data
ball_tree_root = BallTreeNode(data)

# Query with a test point
test_point = np.array([5.1, 3.5, 1.4, 0.2])
neighbors = ball_tree_nearest_neighbor(ball_tree_root, test_point, k=3)
print("Nearest neighbors to the test point are:", neighbors)
