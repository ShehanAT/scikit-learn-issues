import pytest
import itertools
import numpy as np
from numpy.linalg import norm

from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import MultiTaskLassoCV as sklearn_MultiTaskLassoCV
from sklearn.linear_model import MultiTaskLasso as sklearn_MultiTaskLasso
from sklearn.linear_model import lasso_path

from celer import (Lasso, GroupLasso, MultiTaskLasso,
                   MultiTaskLassoCV)
from celer.homotopy import celer_path, mtl_path, _grp_converter
from celer.group_fast import dnorm_grp
from celer.utils.testing import build_dataset

from sklearn.svm import LinearSVC

def test_MultiTaskLasso(fit_intercept):
    """Test that our MultiTaskLasso behaves as sklearn's."""
    X, Y = build_dataset(n_samples=20, n_features=30, n_targets=10)
    alpha_max = np.max(norm(X.T.dot(Y), axis=1)) / X.shape[0]

    alpha = alpha_max / 2.
    params = dict(alpha=alpha, fit_intercept=fit_intercept, tol=1e-10)
    clf = MultiTaskLasso(**params)
    clf.verbose = 2
    clf.fit(X, Y)

    clf2 = sklearn_MultiTaskLasso(**params)
    clf2.fit(X, Y)
    np.testing.assert_allclose(clf.coef_, clf2.coef_, rtol=1e-5)
    if fit_intercept:
        np.testing.assert_allclose(clf.intercept_, clf2.intercept_)

    clf.tol = 1e-7
    check_estimator(clf)
# test_MultiTaskLasso(10);

check_estimator(LinearSVC())

