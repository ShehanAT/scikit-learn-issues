from sklearn.base import BaseEstimator 
from sklearn.utils import check_array, check_X_y
from sklearn.utils.estimator_checks import check_estimator

class DummyEstimator(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        X, y = check_X_y(
            X,
            y,
            accept_sparse = False,
            accept_large_sparse = False,
            ensure_min_features = 3,
            force_all_finite = 'allow-nan')
        return self

    def predict(self, X):
        X  = check_array(
            X,
            accept_sparse = False,
            accept_large_sparse = False,
            # ensure_min_features = 3,
            force_all_finite = 'allow-nan')
        
        # very dumb example but it does not matter
        y_pred = X[:,0] + X[:,1] + X[:,3]
        
        y_pred  = check_array(
            y_pred,
            accept_sparse = False,
            accept_large_sparse = False,
            # ensure_min_features = 3,
            ensure_2d = False,
            force_all_finite = 'allow-nan')
        
        return y_pred

check_estimator(DummyEstimator())