# First, let's install the necessary libraries
!pip install scikit-learn

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree

# Load the dataset
data, target = datasets.load_iris(return_X_y=True)

# Split the data into sets R and S
R = data[:75]
S = data[75:]

# Build B+-tree based iDistance indexes using BallTree for sets R and S
tree_R = BallTree(R)
tree_S = BallTree(S)

# For iJoin
def iJoin(tree_R, tree_S, search_radius):
    indices = tree_R.query_radius(S, r=search_radius, return_distance=False)
    join_pairs = [(R[i], S[j]) for i, inner_list in enumerate(indices) for j in inner_list]
    return join_pairs

# For iJoinAC - using approximations (PCA in this case)
def iJoinAC(tree_R, tree_S, search_radius):
    pca = PCA(n_components=2)
    R_approx = pca.transform(R)
    S_approx = pca.transform(S)

    tree_R_approx = BallTree(R_approx)
    tree_S_approx = BallTree(S_approx)

    indices = tree_R_approx.query_radius(S_approx, r=search_radius, return_distance=False)
    join_pairs = [(R[i], S[j]) for i, inner_list in enumerate(indices) for j in inner_list]
    return join_pairs

# For iJoinDR - using dimensionality reduction (PCA)
def iJoinDR(tree_R, tree_S, search_radius):
    pca = PCA(n_components=1)
    R_reduced = pca.transform(R)
    S_reduced = pca.transform(S)

    tree_R_reduced = BallTree(R_reduced)
    tree_S_reduced = BallTree(S_reduced)

    indices = tree_R_reduced.query_radius(S_reduced, r=search_radius, return_distance=False)
    join_pairs = [(R[i], S[j]) for i, inner_list in enumerate(indices) for j in inner_list]
    return join_pairs

# Run the algorithms
search_radius = 0.5
iJoin_results = iJoin(tree_R, tree_S, search_radius)
iJoinAC_results = iJoinAC(tree_R, tree_S, search_radius)
iJoinDR_results = iJoinDR(tree_R, tree_S, search_radius)

print("iJoin results:", len(iJoin_results))
print("iJoinAC results:", len(iJoinAC_results))
print("iJoinDR results:", len(iJoinDR_results))
