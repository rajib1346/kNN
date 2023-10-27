# First, let's install the necessary libraries
!pip install Rtree
!pip install scikit-learn

# Importing necessary libraries
from rtree import index
from sklearn import datasets
import numpy as np

# Load the dataset
data, target = datasets.load_iris(return_X_y=True)

# Split the data into sets R and S
R = data[:75]
S = data[75:]

# Create the R-tree index for sets R and S
idx_r = index.Index()
idx_s = index.Index()

for i, r in enumerate(R):
    idx_r.insert(i, tuple(r))

for i, s in enumerate(S):
    idx_s.insert(i, tuple(s))

# Sample kNN Join function using R-tree
def knn_join(r_idx, s_idx, k):
    results = []

    for r_id in r_idx.intersection((0, 0, 10, 10), objects=True):
        neighbors = list(s_idx.nearest(r_id.bbox, k))
        results.append((r_id.id, neighbors))

    return results

# Get k nearest neighbors from S for each point in R
k = 5
neighbors = knn_join(idx_r, idx_s, k)

# Sample Output
for r, ns in neighbors[:5]:
    print(f"Point R_{r} neighbors in S: {ns}")
