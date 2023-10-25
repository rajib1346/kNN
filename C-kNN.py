import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

def clustered_knn(X_train, y_train, k_clusters=3, k_neighbors=3):
    # Cluster the data using k-means
    kmeans = KMeans(n_clusters=k_clusters)
    clusters = kmeans.fit_predict(X_train)

    # Use the cluster centers as the new training samples
    new_X_train = kmeans.cluster_centers_
    
    # Assign weight values based on the number of samples in each cluster
    weights = [np.sum(clusters == i) for i in range(k_clusters)]
    weighted_y_train = [np.bincount(y_train[clusters == i]).argmax() for i in range(k_clusters)]
    
    # Train kNN using the new training samples
    knn = KNeighborsClassifier(n_neighbors=k_neighbors, weights='uniform')
    knn.fit(new_X_train, weighted_y_train)
    
    return knn

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the C-kNN model
cknn = clustered_knn(X_train, y_train)

# Predict using the C-kNN model
y_pred = cknn.predict(X_test)

# Calculate accuracy
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
