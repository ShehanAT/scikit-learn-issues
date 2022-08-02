from sklearn.datasets import fetch_lfw_pairs
from sklearn.utils import resample

X = fetch_lfw_pairs(
    subset="test",
    # funneled=False,
    # slice_=(slice(0, 250), slice(0, 250)),
    resize=1,
    color=True
  )

n_samples = 1
imgs, y = resample(X.pairs, X.target, n_samples=n_samples, random_state=101)

print(imgs[0][0])