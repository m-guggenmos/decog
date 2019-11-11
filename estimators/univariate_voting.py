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
            x_sorted = x[ind]
            thresholds = x_sorted[:-1] + np.diff(x_sorted)/2
            comp = (x[None, ...] > thresholds[..., None]).astype(np.int)
            comp[comp] = self.classes_[1]
            comp[np.logical_not(comp)] = self.classes_[0]
            accuracy = np.mean(comp == y, axis=1)
            accuracy_abs = 0.5 + np.abs(accuracy - 0.5)
            best = accuracy_abs.argmax()
            self.threshold_max[f] = thresholds[best]
            self.flip_class_order_max[f] = accuracy[best] < 0.5

            # for i, sample in enumerate(x[ind_pat[:-1]]):
            #     next_sample = x[ind_pat[i + 1]]
            #     threshold = sample + (next_sample - sample) / 2.
            #     pred = np.mean(self.classes_[(x[ind_pat] > threshold).astype(int)] == y[ind_pat])
            #     if pred < 0.5:
            #         pred = 1 - pred
            #         flip_class_order = True
            #     else:
            #         flip_class_order = False
            #     if pred > accuracy_max:
            #         accuracy_max = pred
            #         self.threshold_max[f] = threshold
            #         self.flip_class_order_max[f] = flip_class_order

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
    #         votes = robust_scale(abs(X - self.prototypes_[0, :]) - abs(X - self.prototypes_[1, :]), with_centering=False,
    #                              axis=1)
    #     else:
    #         votes = 2 * (abs(X - self.prototypes_[0, :]) > abs(X - self.prototypes_[1, :])) - 1
    #     if self.dist_weighting is None:
    #         dec = np.sum(votes, 1) / votes.shape[1]
    #     else:
    #         dec = _weighted_sum(votes, self.feature_importances_) / votes.shape[1]
    #     return dec
