from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class UnivariateVoting(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=None, verbose=0, average=False):

        self.random_state = random_state
        self.average = average
        self.verbose = verbose

    def fit(self, X, y):

        self.classes_ = np.unique(y)

        X = np.array(X)
        y = np.array(y)

        if self.average:
            X = np.atleast_2d(np.mean(X, axis=1))
        else:
            X = X.swapaxes(0, 1)

        n_features = len(X)

        self.threshold_max = np.full(n_features, np.nan)
        self.flip_class_order_max = np.full(n_features, np.nan, dtype=np.bool)
        for f, x in enumerate(X):
            accuracy_max = 0
            ind = np.argsort(x)
            for i, sample in enumerate(x[ind[:-1]]):
                next_sample = x[ind[i + 1]]
                threshold = sample + (next_sample - sample) / 2.
                accuracy = np.mean(self.classes_[(x[ind] > threshold).astype(int)] == y[ind])
                if accuracy < 0.5:
                    accuracy = 1 - accuracy
                    flip_class_order = True
                else:
                    flip_class_order = False
                if accuracy > accuracy_max:
                    accuracy_max = accuracy
                    self.threshold_max[f] = threshold
                    self.flip_class_order_max[f] = flip_class_order

        return self

    def predict(self, X):

        X = np.array(X)

        if self.average:
            X = np.expand_dims(np.mean(X, axis=1), axis=1)
        n_samples = len(X)

        predictions = (np.mean([[(b, not b)[self.flip_class_order_max[i]]
                                 for i, b in enumerate(X[s] > self.threshold_max)]
                                for s in range(n_samples)], axis=1) > 0.5).astype(np.int)

        return self.classes_[predictions]

    # def decision_function(self, X):
    #
    #     if self.vote_weighting:
    #         votes = robust_scale(abs(X - self.averages[0, :]) - abs(X - self.averages[1, :]), with_centering=False,
    #                              axis=1)
    #     else:
    #         votes = 2 * (abs(X - self.averages[0, :]) > abs(X - self.averages[1, :])) - 1
    #     if self.dist_weighting is None:
    #         dec = np.sum(votes, 1) / votes.shape[1]
    #     else:
    #         dec = _weighted_sum(votes, self.feature_importances_) / votes.shape[1]
    #     return dec