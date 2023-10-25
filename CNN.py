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

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def simple_knn(X, y, x_query):
    distances = [euclidean_distance(x, x_query) for x in X]
    nearest = np.argmin(distances)
    return y[nearest]

# Step 3: Implement the CNN rule
def condensed_nn(X_train, y_train):
    # Start with a random instance from the dataset
    indices = list(range(len(X_train)))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    X_condensed = [X_train[0]]
    y_condensed = [y_train[0]]
    
    for x, label in zip(X_train, y_train):
        predicted_label = simple_knn(X_condensed, y_condensed, x)
        if predicted_label != label:
            X_condensed.append(x)
            y_condensed.append(label)
    
    return np.array(X_condensed), np.array(y_condensed)

X_condensed, y_condensed = condensed_nn(X_train, y_train)

# Step 4: Use the condensed set for k-NN classification
class KNN:
    def __init__(self, k=1):
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
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

clf = KNN()
clf.fit(X_condensed, y_condensed)
predictions = clf.predict(X_test)

# Evaluate the model
accuracy = np.mean(predictions == y_test)
print(f"Accuracy using Condensed Nearest Neighbor rule: {accuracy * 100:.2f}%")
