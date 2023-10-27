import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import NearestNeighbors
from rtree import index

# Load datasets R and S (for simplicity, we use the same dataset for both)
data = load_iris().data
R = data[:75]
S = data[75:]

# H-BNLJ
def h_bnlj(R, S, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(S)
    distances, indices = nbrs.kneighbors(R)
    results = []
    for rid, (dist, sids) in enumerate(zip(distances, indices)):
        for d, sid in zip(dist, sids):
            results.append((rid, sid + len(R), d))
    return results

# H-BRJ
def h_brj(R, S, k):
    idx = index.Index()
    for sid, s in enumerate(S):
        idx.insert(sid, tuple(s)*2)
    
    results = []
    for rid, r in enumerate(R):
        nearest = list(idx.nearest(tuple(r)*2, k))
        for sid in nearest:
            d = np.linalg.norm(S[sid] - r)
            results.append((rid, sid + len(R), d))
    return results

print("H-BNLJ Results:")
print(h_bnlj(R, S, 3))

print("\nH-BRJ Results:")
print(h_brj(R, S, 3))
