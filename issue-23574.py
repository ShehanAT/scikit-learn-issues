import numpy as np 
from sklearn.cluster import KMeans 
import sklearn 

print("scikit-learn")
print(sklearn.__version__)

print("numpy version")
print(np.__version__)

X_train = np.random.random((10, 2))
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_train)
