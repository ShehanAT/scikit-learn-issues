from sklearn.linear_model import SGDClassifier
from sklearn.utils.estimator_checks import check_estimator


estimator = SGDClassifier(loss="log_loss")
check_estimator(estimator=estimator)