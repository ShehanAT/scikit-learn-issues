from sklearn.neighbors import NearestCentroid
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
nearest_centroid = NearestCentroid(metric="manhattan1")
nearest_centroid.fit(X,y)
nearest_centroid.predict(X)