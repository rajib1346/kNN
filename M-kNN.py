import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier

def find_representative(X, y):
    distances = pairwise_distances(X)
    representatives = []

    for i, xi in enumerate(X):
        # Find the data points with the same class
        same_class_indices = np.where(y == y[i])[0]
        same_class_distances = distances[i, same_class_indices]
        
        # Sort the distances
        sorted_indices = np.argsort(same_class_distances)
        
        # The representative is the data point with maximum distance in the same class neighborhood
        rep_index = same_class_indices[sorted_indices[-1]]
        
        # Store the representative with its distance and count
        representatives.append((rep_index, same_class_distances[sorted_indices[-1]], len(sorted_indices)))
    
    # Deduplicate the representatives
    unique_reps = {r[0]: r for r in representatives}
    
    return list(unique_reps.values())

def model_based_knn(X, y, new_data):
    representatives = find_representative(X, y)
    
    for rep, max_distance, _ in representatives:
        if np.linalg.norm(new_data - X[rep]) <= max_distance:
            return y[rep]

    # If data is not represented, use kNN
    knn = KNeighborsClassifier()
    knn.fit(X, y)
    return knn.predict([new_data])[0]

# Load iris data
iris = load_iris()
X, y = iris.data, iris.target

# Test the Model Based kNN
test_data = X[50]  # Just an example
predicted_class = model_based_knn(X, y, test_data)
print(f"Predicted class for test data: {predicted_class}")
