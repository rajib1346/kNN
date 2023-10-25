import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from scipy.spatial import distance

# Function to find the distance from a point to a line segment
def point_to_segment_distance(p, a, b):
    """Compute distance from point p to segment between a and b."""
    segment_len_sq = np.sum((a - b) ** 2)
    if segment_len_sq == 0:  # a and b are the same point
        return euclidean(p, a)
    t = max(0, min(1, np.dot(p - a, b - a) / segment_len_sq))
    projection = a + t * (b - a)
    return euclidean(p, projection)

def nfl_classify(test_point, trajectories):
    min_distance = float('inf')
    nearest_class = -1
    for class_label, trajectory in trajectories.items():
        for i in range(len(trajectory) - 1):
            d = point_to_segment_distance(test_point, trajectory[i], trajectory[i + 1])
            if d < min_distance:
                min_distance = d
                nearest_class = class_label
    return nearest_class

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create trajectories for each class
# Here, for simplicity, we will order the images in each class by their L2 norm 
# and connect them in that order to form the trajectory
trajectories = {}
for class_label in np.unique(y_train):
    class_data = X_train[y_train == class_label]
    sorted_indices = np.argsort([np.linalg.norm(data) for data in class_data])
    trajectories[class_label] = class_data[sorted_indices]

# Classification using NFL
y_pred = [nfl_classify(test_data, trajectories) for test_data in X_test]

# Calculate accuracy
accuracy = np.mean(np.array(y_pred) == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
