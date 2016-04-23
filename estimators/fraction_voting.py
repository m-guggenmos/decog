from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import ttest_ind
import numpy as np
from sklearn.metrics.classification import _weighted_sum
from sklearn.preprocessing.data import minmax_scale, maxabs_scale, robust_scale
import warnings

WEIGHTING_DICT = {'t': (ttest_ind, {'equal_var': True}), 'welch': (ttest_ind, {'equal_var': False})}

class FractionVoting(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=None, verbose=0):

        pass

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        means_classes = [np.mean(X[y == self.classes_[0]], axis=0),
                         np.mean(X[y == self.classes_[1]], axis=0)]

        self.class_lower = np.sign(np.mean(means_classes[0] < means_classes[1]) - 0.5) == -1
        self.mean_lower = means_classes[self.class_lower]

        return self

    def predict(self, X):

        X = np.array(X)

        dec = self.decision_function(X)

        return self.classes_[(dec > 0).astype(int)]

    def decision_function(self, X):

        tendency = np.mean(X > np.tile(self.mean_lower, (X.shape[0], 1)), axis=1) - 0.5

        dec = tendency * ((-1)**((self.class_lower == self.classes_[0]) + 1))

        return dec