import numpy as np
import pandas as pd
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

rng = np.random.RandomState(0)
n_samples = 100
conn = np.ones([n_samples, n_samples], dtype=bool)
# Add some random sparsity, 70% at most
masked = (rng.random((n_samples, 70)) * n_samples).astype(int)
for i, nbh in enumerate(masked):
    conn[i, nbh] = False
    conn[nbh, i] = False
X = rng.randn(n_samples, 1)
clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=1,
    connectivity=conn,
    linkage='complete',
    affinity='l1',
)
clustering.fit(X)
df = pd.DataFrame({'labels': clustering.labels_, 'x': X.flatten()})
clustered_df = df.groupby('labels').agg(**{op: ('x', op) for op in ['min', 'max']})
# This should be < distance_threshold
print((clustered_df['max'] - clustered_df['min']).max())