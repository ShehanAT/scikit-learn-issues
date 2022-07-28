import numpy as np
from sklearn.neighbors import BallTree

np.random.seed(61)

X = np.random.randint(0, 3, size=(10,2))  # generated dataset

tree = BallTree(X, 4, 'hamming')
distances, indices = tree.query(X, 3, return_distance=True, dualtree=False, breadth_first=False, sort_results=True)

for i, (point, distances, indices) in enumerate(zip(X, distances, indices)):
    print(f'index {i}: datapoint {point} distances: {[round(dist,2) for dist in distances]} indices: {indices}')
