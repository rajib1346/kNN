import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

# Load the iris dataset
data = load_iris()
X = data.data
y = data.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def knn_search(X_train, X_test, k=3):
    distances = pairwise_distances(X_test, X_train)  # Step 1: Compute distances
    k_indices = np.argsort(distances, axis=1)[:, :k]  # Step 2: Sort and select top-k indices
    return k_indices

# Get the k-nearest neighbors for the test set
k_indices = knn_search(X_train, X_test, k=3)

# Print the results
for i, indices in enumerate(k_indices):
    print(f"Test sample {i}:")
    print("Nearest neighbors:", indices)
    print("Corresponding labels:", y_train[indices])
    print("-----------------------------")
