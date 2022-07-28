from sklearn.datasets import load_iris
from sklearn.cluster import Birch

X, y = load_iris(return_X_y=True, as_frame=True)
birch = Birch(n_clusters=3)
birch.fit_predict(X)