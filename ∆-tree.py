from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np

class DeltaTree:
    def __init__(self, n_levels=3):
        self.n_levels = n_levels
        self.levels = []
    
    def fit(self, X):
        # Compute dimensions for each level based on the total number of levels
        dimensions = np.linspace(X.shape[1], 1, self.n_levels, dtype=int)
        
        for d in dimensions:
            pca = PCA(n_components=d)
            transformed_data = pca.fit_transform(X)
            self.levels.append({
                'data': transformed_data,
                'pca': pca,
                'original_data': X
            })
            X = transformed_data
    
    def knn_search(self, query_point, k=1):
        knn_result = None
        knn_distance = float('inf')
        
        for level in self.levels:
            pca = level['pca']
            transformed_query = pca.transform([query_point])[0]
            data = level['data']
            
            distances = np.linalg.norm(data - transformed_query, axis=1)
            closest_idx = np.argmin(distances)
            closest_point_distance = np.linalg.norm(level['original_data'][closest_idx] - query_point)
            
            if closest_point_distance < knn_distance:
                knn_distance = closest_point_distance
                knn_result = level['original_data'][closest_idx]
        
        return knn_result

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data

# Initialize and fit the tree
tree = DeltaTree(n_levels=3)
tree.fit(X)

# Query the tree
query = [5.1, 3.5, 1.4, 0.2]
print(tree.knn_search(query))
