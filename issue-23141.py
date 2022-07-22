import numpy as np
from sklearn.cluster import MiniBatchKMeans

points = [
    [-2636.705, 892.6364, 239.4284], [-2676.219, 922.741, 227.3839], [-2628.628, 902.6482, 245.5609], [-2612.497, 860.9032, 248.924],
    [-2639.552, 993.8482, 211.2253], [-2602.453, 958.7801, 211.5786], [-2598.118, 1032.398, 177.4023], [-2582.155, 972.5088, 203.5048],
    [-2548.377, 803.9934, 279.4388], [-2550.095, 979.9586, 222.6467], [-2746.966, 1021.456, 188.8456], [-2745.181, 984.1931, 199.6674],
    [-2729.113, 973.8251, 201.8876], [-2720.765, 1014.262, 205.0213], [-2747.317, 1099.313, 146.2305], [-2739.32, 1005.173, 200.297]
]

for numClusters in range(7, 17):
    model = MiniBatchKMeans(n_clusters=numClusters, random_state=0, reassignment_ratio=0.0)
    clusters = model.fit_predict(points)

    unique = np.unique(clusters)
    print("requested", str(numClusters).rjust(2), "clusters and result has", str(len(unique)).rjust(2), "clusters with labels", unique)