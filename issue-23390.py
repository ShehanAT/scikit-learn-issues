from sklearn.linear_model import Perceptron
import numpy as np

n_samples = 20
X = np.zeros((n_samples, 1))
y = np.array([x % 2 for x in range(n_samples)])

ppn = Perceptron().fit(X, y)

print("X: ")
print(X)

print("y: ")
print(y)

print("ppn: ")
print(ppn.n_iter_ * n_samples)

print("ppn.t: ")
print(ppn.t_)