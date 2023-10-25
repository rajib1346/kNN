import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

# Step 1: Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataset = pd.read_csv(url, names=names)

# Step 2: Preprocess the data
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Implement weighted k-NN
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class WeightedKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_distances = [distances[i] for i in k_indices]
        k_labels = [self.y_train[i] for i in k_indices]

        # Calculate the weighted votes
        label_weights = {label: 0 for label in set(k_labels)}
        for idx, label in enumerate(k_labels):
            weight = 1 / (k_distances[idx] ** 2 + 1e-5)  # Avoid division by zero
            label_weights[label] += weight

        # Return the label with the highest weight
        return max(label_weights, key=label_weights.get)

# Create k-NN classifier and train it
clf = WeightedKNN(k=3)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Step 4: Evaluate the model
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
