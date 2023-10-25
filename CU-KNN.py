# First, you might need to install these libraries:
# !pip install cudf-cuda110 cuml-cuda110

import numpy as np
import time
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cudf
import cuml

# Generate dataset
X, y = make_classification(n_samples=100000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CPU-based kNN
start_time = time.time()
knn_cpu = KNeighborsClassifier(n_neighbors=5)
knn_cpu.fit(X_train, y_train)
y_pred_cpu = knn_cpu.predict(X_test)
end_time = time.time()

print("CPU kNN Accuracy: ", accuracy_score(y_test, y_pred_cpu))
print("CPU kNN Time: ", end_time - start_time, "seconds")

# Convert data to GPU-based dataframes
X_train_gpu = cudf.DataFrame.from_pandas(pd.DataFrame(X_train))
X_test_gpu = cudf.DataFrame.from_pandas(pd.DataFrame(X_test))
y_train_gpu = cudf.Series(y_train)

# GPU-based kNN
start_time = time.time()
knn_gpu = cuml.neighbors.KNeighborsClassifier(n_neighbors=5)
knn_gpu.fit(X_train_gpu, y_train_gpu)
y_pred_gpu = knn_gpu.predict(X_test_gpu)
end_time = time.time()

print("GPU CU-kNN Accuracy: ", accuracy_score(y_test, y_pred_gpu.to_array()))
print("GPU CU-kNN Time: ", end_time - start_time, "seconds")
