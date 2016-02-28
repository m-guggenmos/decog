from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import ttest_ind
import numpy as np
from sklearn.metrics.classification import _weighted_sum
from sklearn.preprocessing.data import minmax_scale, maxabs_scale, robust_scale
import warnings

WEIGHTING_DICT = {'t': (ttest_ind, {'equal_var': True}), 'welch': (ttest_ind, {'equal_var': False})}

class WeightedVoting(BaseEstimator, ClassifierMixin):
    def __init__(self, vote_weighting=True, dist_weighting='t', averaging='median', random_state=None, verbose=0):

        self.vote_weighting = vote_weighting
        self.dist_weighting = dist_weighting
        self.averaging = averaging
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        self.classes_ = np.unique(y)

        X = np.array(X)

        dist_weighting_ = 't' if self.dist_weighting is None else self.dist_weighting
        w = WEIGHTING_DICT[dist_weighting_]
        with warnings.catch_warnings():  # (1 / 2) we catch the warning about nans here
            warnings.simplefilter("ignore", RuntimeWarning)
            statistic = w[0](X[np.array(y) == self.classes_[0], :], X[np.array(y) == self.classes_[1], :], **w[1]).statistic
        statistic[np.isnan(statistic)] = 0  # (2 / 2) and clean up the mess afterwards
        # self.feature_importances_ = np.atleast_1d(robust_scale(abs(statistic)[:, np.newaxis]).squeeze())
        self.feature_importances_ = np.atleast_1d(abs(statistic[:, np.newaxis]).squeeze())

        adict = {'median': np.median, 'mean': np.mean}
        self.averages = np.vstack((adict[self.averaging](X[np.array(y) == self.classes_[0], :], 0),
                                   adict[self.averaging](X[np.array(y) == self.classes_[1], :], 0)))

        return self

    def predict(self, X):

        # if self.vote_weighting:
        #     votes = maxabs_scale(abs(X - self.averages[0, :]) - abs(X - self.averages[1, :]))
        # else:
        #     votes = 2 * (abs(X - self.averages[0, :]) > abs(X - self.averages[1, :])) - 1
        # if self.dist_weighting is None:
        #     indices = (np.sum(votes, 1) > 0).astype(int)
        # else:
        #     indices = (_weighted_sum(votes, self.weights) > 0).astype(int)
        # return self.classes_[indices]

        dec = self.decision_function(X)
        return self.classes_[(dec > 0).astype(int)]

    def decision_function(self, X):

        if self.vote_weighting:
            votes = robust_scale(abs(X - self.averages[0, :]) - abs(X - self.averages[1, :]), with_centering=False,
                                 axis=1)
        else:
            votes = 2 * (abs(X - self.averages[0, :]) > abs(X - self.averages[1, :])) - 1
        if self.dist_weighting is None:
            dec = np.sum(votes, 1) / votes.shape[1]
        else:
            dec = _weighted_sum(votes, self.feature_importances_) / votes.shape[1]
        return dec