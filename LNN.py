import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from numpy.linalg import norm

def point_to_line_distance(point, line_points):
    """Calculate the distance from point to the line formed by line_points."""
    a, b = line_points
    normal_vector = np.cross(a - point, b - point)
    distance = norm(normal_vector) / norm(b - a)
    return distance

def lnn_classify(test_point, X_train, y_train):
    unique_classes = np.unique(y_train)
    min_distance = float('inf')
    nearest_class = -1
    
    for uc in unique_classes:
        class_points = X_train[y_train == uc]
        distances = [euclidean(test_point, cp) for cp in class_points]
        nearest_indices = np.argsort(distances)[:2]
        distance_to_line = point_to_line_distance(test_point, class_points[nearest_indices])
        
        if distance_to_line < min_distance:
            min_distance = distance_to_line
            nearest_class = uc
            
    return nearest_class

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classification using LNN
y_pred = [lnn_classify(test_data, X_train, y_train) for test_data in X_test]

# Calculate accuracy
accuracy = np.mean(np.array(y_pred) == y_test)
print(f"Accuracy using NNL: {accuracy * 100:.2f}%")
