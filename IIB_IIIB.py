# First, let's install the necessary libraries
!pip install scikit-learn

from sklearn.datasets import fetch_20newsgroups_vectorized
from collections import defaultdict

# Load the dataset
data = fetch_20newsgroups_vectorized(subset='all')

# For simplicity, take a subset
Br = data.data[:500]
Bs = data.data[500:1000]

# BF Algorithm
def BF(Br, Bs):
    neighbors = defaultdict(list)
    for i, r in enumerate(Br):
        pruning_score = 0
        for j, s in enumerate(Bs):
            similarity = r.dot(s.T).toarray()[0][0]
            if similarity > pruning_score:
                neighbors[i].append(j)
                pruning_score = similarity
    return neighbors

# IIB Algorithm
def IIB(Br, Bs):
    # Create inverted index
    inverted_index = defaultdict(list)
    for j, s in enumerate(Bs):
        for index in s.indices:
            inverted_index[index].append(j)

    neighbors = defaultdict(list)
    for i, r in enumerate(Br):
        candidates = set()
        for index in r.indices:
            candidates.update(inverted_index[index])
        pruning_score = 0
        for j in candidates:
            similarity = r.dot(Bs[j].T).toarray()[0][0]
            if similarity > pruning_score:
                neighbors[i].append(j)
                pruning_score = similarity
    return neighbors

# IIIB Algorithm
def IIIB(Br, Bs):
    # Create inverted index as in IIB
    inverted_index = defaultdict(list)
    for j, s in enumerate(Bs):
        for index in s.indices:
            inverted_index[index].append(j)

    neighbors = defaultdict(list)
    previous_threshold = 0
    for i, r in enumerate(Br):
        candidates = set()
        for index in r.indices:
            candidates.update(inverted_index[index])
        pruning_score = previous_threshold
        for j in candidates:
            similarity = r.dot(Bs[j].T).toarray()[0][0]
            if similarity > pruning_score:
                neighbors[i].append(j)
                pruning_score = similarity
        previous_threshold = pruning_score
    return neighbors

# Execute the algorithms
neighbors_BF = BF(Br, Bs)
neighbors_IIB = IIB(Br, Bs)
neighbors_IIIB = IIIB(Br, Bs)

# Print the number of neighbors found for the first 5 queries
for i in range(5):
    print(f"Query {i} - BF: {len(neighbors_BF[i])}, IIB: {len(neighbors_IIB[i])}, IIIB: {len(neighbors_IIIB[i])}")
