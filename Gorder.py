# First, let's install the necessary libraries
!pip install scikit-learn

import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets

# Load the dataset
data, target = datasets.load_iris(return_X_y=True)

# Phase 1: PCA to find the direction of highest variance
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# Phase 2: Grid Order Sorting (simplified)
# Here we'll use the min-max values from PCA and create a grid

x_max, y_max = data_pca.max(axis=0)
x_min, y_min = data_pca.min(axis=0)

# Assume we split the dataspace into 10x10 grid
grid_size = 10
x_step = (x_max - x_min) / grid_size
y_step = (y_max - y_min) / grid_size

grid_order = []

for point in data_pca:
    x_idx = int((point[0] - x_min) / x_step)
    y_idx = int((point[1] - y_min) / y_step)
    grid_order.append((x_idx, y_idx))

# Sort by grid order
sorted_indexes = sorted(range(len(grid_order)), key=lambda k: grid_order[k])
data_sorted = data[sorted_indexes]

# Phase 3: Scheduled Block Nested Loop Join on G-ordered data (simplified version)
BLOCK_SIZE = 25  # As per the experimental result
pruning_distance = 0.5  # A sample value, this should be determined based on your dataset and requirements

results = []

for i in range(0, len(data_sorted), BLOCK_SIZE):
    block_r = data_sorted[i:i+BLOCK_SIZE]
    for j in range(0, len(data_sorted), BLOCK_SIZE):
        block_s = data_sorted[j:j+BLOCK_SIZE]
        
        min_distance = np.min([np.linalg.norm(r - s) for r in block_r for s in block_s])
        
        if min_distance < pruning_distance:
            for r in block_r:
                neighbors = sorted(block_s, key=lambda s: np.linalg.norm(r - s))[:10]  # 10-nearest neighbors as an example
                results.append((r, neighbors))

# Sample output
for r, neighbors in results[:5]:
    print(f"Point {r} neighbors: {neighbors}")
