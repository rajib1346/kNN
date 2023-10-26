import numpy as np
from sklearn import datasets

# 1. Cantor Pairing Function
def cantor_pair(k1, k2):
    return 0.5 * (k1 + k2) * (k1 + k2 + 1) + k2

# 2. PL-Tree Approach
class PLTree:
    def __init__(self, limit):
        self.limit = limit
        self.labels = []
    
    def partition_and_label(self, data):
        # Base condition: If data points are less than or equal to limit, label them
        if len(data) <= self.limit:
            k1, k2 = data[:, 0].mean(), data[:, 1].mean()  # Averaging for simplicity (can be replaced by another representative point)
            label = cantor_pair(k1, k2)
            self.labels.extend([label] * len(data))
            return
        
        # Else, split the data and recursively label
        median = np.median(data, axis=0)
        lower_left = data[(data[:, 0] < median[0]) & (data[:, 1] < median[1])]
        lower_right = data[(data[:, 0] >= median[0]) & (data[:, 1] < median[1])]
        upper_left = data[(data[:, 0] < median[0]) & (data[:, 1] >= median[1])]
        upper_right = data[(data[:, 0] >= median[0]) & (data[:, 1] >= median[1])]
        
        self.partition_and_label(lower_left)
        self.partition_and_label(lower_right)
        self.partition_and_label(upper_left)
        self.partition_and_label(upper_right)
    
    def fit(self, data):
        self.partition_and_label(data)
        return self.labels

# 3. Testing the PL-Tree using iris dataset
data = datasets.load_iris().data[:, :2]  # Only taking 2 dimensions for simplicity
pltree = PLTree(limit=10)
labels = pltree.fit(data)
print(labels)
