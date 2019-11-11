from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from scipy.stats import pearsonr, rankdata
from scipy.stats.stats import _sum_of_squares
from scipy.spatial.distance import euclidean, correlation, mahalanobis, cityblock
from itertools import product, combinations
from scipy.linalg import fractional_matrix_power
from sklearn.discriminant_analysis import _cov as sklearn_cov
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, _class_cov, _class_means, linalg
from scipy.linalg import pinv, pinv2

class DVWrapper(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, estimator=None, random_state=None, verbose=0):

        self.estimator = estimator
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self.estimator.fit(X, y)

    def predict(self, X, y=None):

        if hasattr(self.estimator, 'decision_function'):
            return np.abs(self.estimator.decision_function(X))
        elif hasattr(self.estimator, 'predict_proba'):
            return self.estimator.predict_proba(X).max(axis=1) - 0.5
        else:
            raise ValueError('Estimator has neither decision_function() nor predict_proba()')

class DV2Wrapper(BaseEstimator):

    _estimator_type = 'DV'

    def __init__(self, estimator=None, random_state=None, verbose=0):

        self.estimator = estimator
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self.estimator.fit(X, y)

    def predict(self, X, y=None):

        if hasattr(self.estimator, 'decision_function'):
            return self.estimator.decision_function(X)
        elif hasattr(self.estimator, 'predict_proba'):
            # if not all(self.estimator.predict_proba(X).max(axis=1) == np.array([1, 1])):
            #     print(self.estimator.predict_proba(X).max(axis=1))
            return (self.estimator.predict_proba(X).max(axis=1) - 0.5)*np.sign(self.estimator.predict_proba(X)[:, 1] - 0.5)
        else:
            raise ValueError('Estimator has neither decision_function() nor predict_proba()')


class AUCWrapper(BaseEstimator):

    _estimator_type = 'AUC'

    def __init__(self, estimator=None, random_state=None, verbose=0):

        self.estimator = estimator
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self.estimator.fit(X, y)

    def predict(self, X, y=None):

        if hasattr(self.estimator, 'decision_function'):
            return self.estimator.decision_function(X)
        elif hasattr(self.estimator, 'predict_proba'):
            # if not all(self.estimator.predict_proba(X).max(axis=1) == np.array([1, 1])):
            #     print(self.estimator.predict_proba(X).max(axis=1))
            return (self.estimator.predict_proba(X).max(axis=1) - 0.5)*np.sign(self.estimator.predict_proba(X)[:, 1] - 0.5)
        else:
            raise ValueError('Estimator has neither decision_function() nor predict_proba()')

class Pearson(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0, return_1d=False):

        self.random_state = random_state
        self.verbose = verbose
        self.return_1d = return_1d

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        d = correlation(np.mean(X[y == self.classes_[0]], axis=0),
                        np.mean(X[y == self.classes_[1]], axis=0))

        if self.return_1d:
            d = np.atleast_1d(d)

        return d

class PearsonWhite(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, whiten=True, sigma=None, fractional_sigma=None, shrinkage='auto', sigma_method='full2', verbose=0):

        self.whiten = whiten
        self.sigma = sigma
        self.fractional_sigma = fractional_sigma
        self.shrinkage = shrinkage
        self.sigma_method = sigma_method

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        d = correlation(np.mean(X[y == self.classes_[0]], axis=0),
                        np.mean(X[y == self.classes_[1]], axis=0))
        return d


class CvPearson(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0, regularize_var=True, regularize_denom=True,
                 reg_factor_var=0.1, reg_factor_denom=0.25, bounded=True, reg_bounding=1,
                 return_1d=False):

        self.random_state = random_state
        self.verbose = verbose
        self.regularize_var = regularize_var
        self.regularize_denom = regularize_denom
        self.reg_factor_var = reg_factor_var
        self.reg_factor_denom = reg_factor_denom
        self.bounded = bounded
        self.reg_bounding = reg_bounding
        self.return_1d = return_1d

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        self.A1 = np.mean(X[y == self.classes_[0]], axis=0)
        self.B1 = np.mean(X[y == self.classes_[1]], axis=0)
        self.var_A1 = np.var(self.A1)
        self.var_B1 = np.var(self.B1)
        self.denom_noncv = np.sqrt(self.var_A1 * self.var_B1)

        return self

    def predict(self, X, y=None):

        X = np.array(X)

        A2 = np.mean(X[y == self.classes_[0]], axis=0)
        B2 = np.mean(X[y == self.classes_[1]], axis=0)

        cov_a1b2 = np.cov(self.A1, B2)[0, 1]
        cov_b1a2 = np.cov(self.B1, A2)[0, 1]
        cov_ab = (cov_a1b2 + cov_b1a2) / 2

        var_A12 = np.cov(self.A1, A2)[0, 1]
        var_B12 = np.cov(self.B1, B2)[0, 1]

        if self.regularize_var:
            # denom = np.sqrt(min(max(self.reg_factor_var * self.var_A1, var_A12), self.var_A1 / self.reg_factor_var) *
            #                 min(max(self.reg_factor_var * self.var_B1, var_B12), self.var_B1 / self.reg_factor_var))
            denom = np.sqrt(max(self.reg_factor_var * self.var_A1, var_A12) * max(self.reg_factor_var * self.var_B1, var_B12))
        else:
            denom = np.sqrt(var_A12 * var_B12)
        if self.regularize_denom:
            # denom = min(max(self.reg_factor_denom * self.denom_noncv, denom), self.denom_noncv / self.reg_factor_denom)
            denom = max(self.reg_factor_denom * self.denom_noncv, denom)


        r = cov_ab / denom

        if self.bounded:
            r = min(max(-self.reg_bounding, r), self.reg_bounding)

        d = 1 - r
        if self.return_1d:
            d = np.atleast_1d(d)
        return d


class CvPearson2(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0, bounded=True, return_1d=False):

        self.bounded = bounded

        self.random_state = random_state
        self.verbose = verbose
        self.return_1d = return_1d

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        self.A1 = np.mean(X[y == self.classes_[0]], axis=0)
        self.B1 = np.mean(X[y == self.classes_[1]], axis=0)
        self.denom = np.sqrt(np.var(self.A1) * np.var(self.B1))

        return self

    def predict(self, X, y=None):

        X = np.array(X)

        A2 = np.mean(X[y == self.classes_[0]], axis=0)
        B2 = np.mean(X[y == self.classes_[1]], axis=0)

        r = (np.cov(self.A1, B2)[0, 1] + np.cov(self.B1, A2)[0, 1]) / (2 * self.denom)

        if self.bounded:
            r = min(max(-1, r), 1)

        d = 1 - r

        if self.return_1d:
            d = np.atleast_1d(d)

        return d


class CvPearsonNoDenom(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        self.A1 = np.mean(X[y == self.classes_[0]], axis=0)
        self.B1 = np.mean(X[y == self.classes_[1]], axis=0)

        return self

    def predict(self, X, y):

        X = np.array(X)

        A2 = np.mean(X[y == self.classes_[0]], axis=0)
        B2 = np.mean(X[y == self.classes_[1]], axis=0)

        d = (np.cov(self.A1, B2)[0, 1] + np.cov(self.B1, A2)[0, 1]) / 2

        return d

class CvPearson3(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0, mincov=1e-3, mindenom=None, bounded=True):

        self.random_state = random_state
        self.verbose = verbose
        self.mincov = mincov
        self.mindenom = mindenom
        self.negbound = bounded

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        self.A1 = np.mean(X[y == self.classes_[0]], axis=0)
        self.B1 = np.mean(X[y == self.classes_[1]], axis=0)

        return self

    def predict(self, X, y):

        X = np.array(X)

        A2 = np.mean(X[y == self.classes_[0]], axis=0)
        B2 = np.mean(X[y == self.classes_[1]], axis=0)

        denom = np.sqrt(max(A2.shape[0]*self.mincov, self.A1@A2) * max(A2.shape[0]*self.mincov, self.B1@B2))
        if self.mindenom is not None and denom < self.mindenom:
            denom = self.mindenom

        r = (self.A1@B2 + self.B1@A2) / (2 * denom)

        if self.negbound:
            r = min(max(-1, r), 1)

        d = 1 - r
        return d



class PearsonBE(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        return np.mean([correlation(p[0], p[1]) for p in
                        product(X[y == self.classes_[0]], X[y == self.classes_[1]])])

class PearsonWI(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y=None):

        X = np.array(X)
        self.classes_ = np.unique(y)

        return np.mean([correlation(X[c[0]], X[c[1]]) for c in combinations(range(X.shape[0]), 2)])

class PearsonWC(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        X1 = X[y == self.classes_[0]]
        X2 = X[y == self.classes_[1]]

        wc = (np.mean([correlation(X1[c[0]], X1[c[1]]) for c in combinations(range(X1.shape[0]), 2)]) +
              np.mean([correlation(X2[c[0]], X2[c[1]]) for c in combinations(range(X2.shape[0]), 2)])) / 2

        return np.mean([correlation(p[0], p[1]) for p in product(X1, X2)]) - wc


class PearsonWC2(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        X1 = X[y == self.classes_[0]]
        X2 = X[y == self.classes_[1]]

        wc = (np.mean([correlation(X1[c[0]], X1[c[1]]) for c in combinations(range(X1.shape[0]), 2)]) +
              np.mean([correlation(X2[c[0]], X2[c[1]]) for c in combinations(range(X2.shape[0]), 2)])) / 2

        return correlation(np.mean(X1, axis=0), np.mean(X2, axis=0)) - wc


class CvPearsonClassifier(BaseEstimator, ClassifierMixin):

    _estimator_type = 'classifier'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        self.A1 = np.mean(X[y == self.classes_[0]], axis=0)
        self.B1 = np.mean(X[y == self.classes_[1]], axis=0)

        return self

    def predict(self, X):

        X = np.array(X)

        predictions = [self.classes_[int(pearsonr(self.B1, x)[0] > pearsonr(self.A1, x)[0])] for x in X]

        return predictions


class CvSpearman(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0, mincov=1e-3, mindenom=None, bounded=True):

        self.random_state = random_state
        self.verbose = verbose
        self.mincov = mincov
        self.mindenom = mindenom
        self.negbound = bounded

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)
        X[y == self.classes_[0]] = [rankdata(x) for x in X[y == self.classes_[0]]]
        X[y == self.classes_[1]] = [rankdata(x) for x in X[y == self.classes_[1]]]

        self.A1 = np.mean(X[y == self.classes_[0]], axis=0)
        self.B1 = np.mean(X[y == self.classes_[1]], axis=0)

        return self

    def predict(self, X, y):

        X = np.array(X)
        X[y == self.classes_[0]] = [rankdata(x) for x in X[y == self.classes_[0]]]
        X[y == self.classes_[1]] = [rankdata(x) for x in X[y == self.classes_[1]]]

        A2 = np.mean(X[y == self.classes_[0]], axis=0)
        B2 = np.mean(X[y == self.classes_[1]], axis=0)

        denom = np.sqrt(max(self.mincov, np.cov(self.A1, A2)[0, 1]) * max(self.mincov, np.cov(self.B1, B2)[0, 1]))
        if self.mindenom is not None and denom < self.mindenom:
            denom = self.mindenom

        r = (np.cov(self.A1, B2)[0, 1] + np.cov(self.B1, A2)[0, 1]) / (2 * denom)

        if self.negbound:
            r = min(max(-1, r), 1)

        return 1 - r


class CvSpearman2(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)
        X[y == self.classes_[0]] = [rankdata(x) for x in X[y == self.classes_[0]]]
        X[y == self.classes_[1]] = [rankdata(x) for x in X[y == self.classes_[1]]]

        self.A1 = np.mean(X[y == self.classes_[0]], axis=0)
        self.B1 = np.mean(X[y == self.classes_[1]], axis=0)
        self.denom = np.sqrt(np.cov(self.A1) * np.cov(self.B1))

        return self

    def predict(self, X, y):

        X = np.array(X)
        X[y == self.classes_[0]] = [rankdata(x) for x in X[y == self.classes_[0]]]
        X[y == self.classes_[1]] = [rankdata(x) for x in X[y == self.classes_[1]]]

        A2 = np.mean(X[y == self.classes_[0]], axis=0)
        B2 = np.mean(X[y == self.classes_[1]], axis=0)

        r = (np.cov(self.A1, B2)[0, 1] + np.cov(self.B1, A2)[0, 1]) / (2 * self.denom)

        return 1 - r


class Spearman(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)
        X[y == self.classes_[0]] = [rankdata(x) for x in X[y == self.classes_[0]]]
        X[y == self.classes_[1]] = [rankdata(x) for x in X[y == self.classes_[1]]]

        return correlation(np.mean(X[y == self.classes_[0]], axis=0),
                           np.mean(X[y == self.classes_[1]], axis=0))

class SpearmanBE(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)
        X[y == self.classes_[0]] = [rankdata(x) for x in X[y == self.classes_[0]]]
        X[y == self.classes_[1]] = [rankdata(x) for x in X[y == self.classes_[1]]]

        return np.mean([correlation(p[0], p[1]) for p in
                        product(X[y == self.classes_[0]], X[y == self.classes_[1]])])

class SpearmanWI(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)
        X[y == self.classes_[0]] = [rankdata(x) for x in X[y == self.classes_[0]]]

        return np.mean([correlation(X[c[0]], X[c[1]]) for c in combinations(range(X.shape[0]), 2)])



class SpearmanWC(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        X1 = np.array([rankdata(x) for x in X[y == self.classes_[0]]])
        X2 = np.array([rankdata(x) for x in X[y == self.classes_[1]]])

        wc = (np.mean([correlation(X1[c[0]], X1[c[1]]) for c in combinations(range(X1.shape[0]), 2)]) +
              np.mean([correlation(X2[c[0]], X2[c[1]]) for c in combinations(range(X2.shape[0]), 2)])) / 2

        return np.mean([correlation(p[0], p[1]) for p in product(X1, X2)]) - wc

class SpearmanWC2(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        X1 = np.array([rankdata(x) for x in X[y == self.classes_[0]]])
        X2 = np.array([rankdata(x) for x in X[y == self.classes_[1]]])

        wc = (np.mean([correlation(X1[c[0]], X1[c[1]]) for c in combinations(range(X1.shape[0]), 2)]) +
              np.mean([correlation(X2[c[0]], X2[c[1]]) for c in combinations(range(X2.shape[0]), 2)])) / 2

        return correlation(np.mean(X1, axis=0), np.mean(X2, axis=0)) - wc


class CvChebyshev(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        self.dist_train = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)

        return self

    def fit_within(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        X0, X1 = X[y == self.classes_[0]], X[y == self.classes_[1]]

        self.dist_train_within = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)

        return self

    def predict(self, X, y):

        X = np.array(X)
        y = np.array(y)

        dist_test = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)

        return self.dist_train @ dist_test


class CvEuclidean(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        self.dist_train = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)

        return self

    def predict(self, X, y):

        X = np.array(X)
        y = np.array(y)

        dist_test = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)

        return self.dist_train @ dist_test


class Manhattan(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        return cityblock(np.mean(X[y == self.classes_[0]], axis=0),
                         np.mean(X[y == self.classes_[1]], axis=0))


class CvManhattan(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        self.dist_train = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)

        return self

    def predict(self, X, y):

        X = np.array(X)
        y = np.array(y)

        dist_test = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)

        return self.dist_train @ dist_test

class CvSSQ(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        self.dist_train = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)
        self.dist_train2 = np.mean(X[y == self.classes_[0]], axis=0) + np.mean(X[y == self.classes_[1]], axis=0)

        return self

    def predict(self, X, y):

        X = np.array(X)
        y = np.array(y)

        dist_test = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)
        dist_test2 = np.mean(X[y == self.classes_[0]], axis=0) + np.mean(X[y == self.classes_[1]], axis=0)

        # return 1 - (self.dist_train @ dist_test) / (self.dist_train2 @ dist_test2)
        return 1 - (self.dist_train @ dist_test) / np.sqrt(np.sum(self.dist_train2**2 + dist_test2**2))


class CvEuclideanX(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X = np.array(X)

        self.dist_train = [X[c[0]] - X[c[1]] for c in combinations(range(X.shape[0]), 2)]

        return self

    def predict(self, X, y):

        X = np.array(X)
        y = np.array(y)

        dist_test = [X[c[0]] - X[c[1]] for c in combinations(range(X.shape[0]), 2)]

        return np.mean([p[0] @ p[1] for p in product(self.dist_train, dist_test)])


class CvEuclideanY(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        try:
            self.dist_train = [p[0] - p[1] for p in product(X[y == self.classes_[0]], X[y == self.classes_[1]])]
        except:
            pass

        return self

    def predict(self, X, y):

        X = np.array(X)
        y = np.array(y)

        dist_test = [p[0] - p[1] for p in product(X[y == self.classes_[0]], X[y == self.classes_[1]])]

        return np.mean([p[0] @ p[1] for p in product(self.dist_train, dist_test)])


class CvEuclideanClassifier(BaseEstimator, ClassifierMixin):

    _estimator_type = 'classifier'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        self.A1 = np.mean(X[y == self.classes_[0]], axis=0)
        self.B1 = np.mean(X[y == self.classes_[1]], axis=0)

        return self

    def predict(self, X):

        X = np.array(X)

        dec = self.decision_value(X)

        predictions = np.array([self.classes_[int(d > 0)] for d in dec])

        return predictions

    def decision_value(self, X):

        X = np.array(X)

        return np.array([euclidean(self.A1, x) - euclidean(self.B1, x) for x in X])

class Euclidean(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        return euclidean(np.mean(X[y == self.classes_[0]], axis=0),
                         np.mean(X[y == self.classes_[1]], axis=0))


class Euclidean2(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        return euclidean(np.mean(X[y == self.classes_[0]], axis=0),
                         np.mean(X[y == self.classes_[1]], axis=0))**2


class EuclideanBE(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        return np.mean([euclidean(p[0], p[1]) for p in
                        product(X[y == self.classes_[0]], X[y == self.classes_[1]])])


class EuclideanWI(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y=None):

        X = np.array(X)
        self.classes_ = np.unique(y)

        return np.mean([euclidean(X[c[0]], X[c[1]]) for c in combinations(range(X.shape[0]), 2)])


class EuclideanWC(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        X1 = X[y == self.classes_[0]]
        X2 = X[y == self.classes_[1]]

        wc = (np.mean([euclidean(X1[c[0]], X1[c[1]]) for c in combinations(range(X1.shape[0]), 2)]) +
              np.mean([euclidean(X2[c[0]], X2[c[1]]) for c in combinations(range(X2.shape[0]), 2)])) / 2

        return np.mean([euclidean(p[0], p[1]) for p in product(X1, X2)]) - wc


class Euclidean2BE(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        return np.mean([euclidean(p[0], p[1])**2 for p in
                        product(X[y == self.classes_[0]], X[y == self.classes_[1]])])


class Euclidean2WI(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y=None):

        X = np.array(X)
        self.classes_ = np.unique(y)

        return np.mean([euclidean(X[c[0]], X[c[1]])**2 for c in combinations(range(X.shape[0]), 2)])


class Euclidean2WC(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        X1 = X[y == self.classes_[0]]
        X2 = X[y == self.classes_[1]]

        wc = (np.mean([euclidean(X1[c[0]], X1[c[1]])**2 for c in combinations(range(X1.shape[0]), 2)]) +
              np.mean([euclidean(X2[c[0]], X2[c[1]])**2 for c in combinations(range(X2.shape[0]), 2)])) / 2

        return np.mean([euclidean(p[0], p[1])**2 for p in product(X1, X2)]) - wc



class CvMahalanobis(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, sigma=None, inv_sigma=None, sigma_method='full2', shrinkage=None, random_state=None, verbose=0):

        self.sigma = sigma
        self.inv_sigma = inv_sigma
        self.sigma_method = sigma_method
        self.shrinkage = shrinkage
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        if self.sigma is None:
            self.sigma = _class_cov(X, y, shrinkage=self.shrinkage)
        if self.inv_sigma is None:
            self.inv_sigma = np.linalg.inv(self.sigma)

        self.dist_train = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)

        return self

    def predict(self, X, y):

        X = np.array(X)
        y = np.array(y)

        dist_test = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)

        return self.dist_train@self.inv_sigma@dist_test

class CvPMahalanobis(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, sigma=None, pinv_sigma=None, sigma_method='full2', shrinkage=None, random_state=None, verbose=0):

        self.sigma = sigma
        self.pinv_sigma = pinv_sigma
        self.sigma_method = sigma_method
        self.shrinkage = shrinkage
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        if self.sigma is None:
            self.sigma = _class_cov(X, y, shrinkage=self.shrinkage)
        if self.pinv_sigma is None:
            self.pinv_sigma = pinv(self.sigma)

        self.dist_train = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)

        return self

    def predict(self, X, y):

        X = np.array(X)
        y = np.array(y)

        dist_test = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)

        return self.dist_train@self.pinv_sigma@dist_test

class CvP2Mahalanobis(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, sigma=None, pinv2_sigma=None, sigma_method='full2', shrinkage=None, random_state=None, verbose=0):

        self.sigma = sigma
        self.pinv2_sigma = pinv2_sigma
        self.sigma_method = sigma_method
        self.shrinkage = shrinkage
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        if self.sigma is None:
            self.sigma = _class_cov(X, y, shrinkage=self.shrinkage)
        if self.pinv2_sigma is None:
            self.pinv2_sigma = pinv2(self.sigma)

        self.dist_train = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)

        return self

    def predict(self, X, y):

        X = np.array(X)
        y = np.array(y)

        dist_test = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)

        return self.dist_train@self.pinv2_sigma@dist_test

class CvMahalanobis2(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, sigma=None, shrinkage=None, random_state=None, verbose=0):

        self.sigma = sigma
        self.shrinkage = shrinkage
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        if self.sigma is None:
            # self.sigma = (np.cov(X[y == self.classes_[0]], rowvar=0) + np.cov(X[y == self.classes_[1]], rowvar=0))/2
            self.sigma = _class_cov(X, y, shrinkage=self.shrinkage)

        self.dist_train = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)

        return self

    def predict(self, X, y):

        X = np.array(X)
        y = np.array(y)

        dist_test = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)

        return self.dist_train@self.sigma@dist_test

class LDC(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, sigma=None, fractional_sigma=None, shrinkage=None,
                 random_state=None, verbose=0):

        self.sigma = sigma
        self.fractional_sigma = fractional_sigma
        self.shrinkage = shrinkage

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        if self.sigma is None:
            # self.sigma = (np.cov(X[y == self.classes_[0]], rowvar=0) + np.cov(X[y == self.classes_[1]], rowvar=0))/2
            self.sigma = _class_cov(X, y, shrinkage=self.shrinkage)
        if self.fractional_sigma is None:
            self.fractional_sigma = fractional_matrix_power(self.sigma, -0.5)

        X = X @ self.fractional_sigma

        self.dist_train = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)

        return self

    def predict(self, X, y):

        X = np.array(X)
        y = np.array(y)

        X = X @ self.fractional_sigma

        dist_test = np.mean(X[y == self.classes_[0]], axis=0) - np.mean(X[y == self.classes_[1]], axis=0)

        return self.dist_train@self.sigma@dist_test


class Mahalanobis(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, sigma=None, inv_sigma=None, sigma_method='full2', shrinkage=None, random_state=None, verbose=0):

        self.sigma = sigma
        self.inv_sigma = inv_sigma
        self.sigma_method = sigma_method
        self.shrinkage = shrinkage

        self.sigma = None

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        # X = X[:, :3]

        # sigma = (np.cov(X[y == self.classes_[0]], rowvar=0) + np.cov(X[y == self.classes_[1]], rowvar=0))/2

        if self.sigma is None:
            self.sigma = _class_cov(X, y, shrinkage=self.shrinkage)
        if self.inv_sigma is None:
            self.inv_sigma = np.linalg.inv(self.sigma)

        # cdist(X[y == self.classes_[0]], X[y == self.classes_[1]], 'mahalanobis')
        # print([mahalanobis(a, beta, np.linalg.inv(sigma)) for a, beta in zip(X[y == self.classes_[0]], X[y == self.classes_[1]])])

        return mahalanobis(np.mean(X[y == self.classes_[0]], axis=0),
                           np.mean(X[y == self.classes_[1]], axis=0),
                           self.inv_sigma)

class PMahalanobis(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, sigma=None, pinv_sigma=None, sigma_method='full2', shrinkage=None, random_state=None, verbose=0):

        self.sigma = sigma
        self.pinv_sigma = pinv_sigma
        self.sigma_method = sigma_method
        self.shrinkage = shrinkage

        self.sigma = None

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        # X = X[:, :3]

        # sigma = (np.cov(X[y == self.classes_[0]], rowvar=0) + np.cov(X[y == self.classes_[1]], rowvar=0))/2

        if self.sigma is None:
            self.sigma = _class_cov(X, y, shrinkage=self.shrinkage)
        if self.pinv_sigma is None:
            self.pinv_sigma = pinv2(self.sigma)

        # cdist(X[y == self.classes_[0]], X[y == self.classes_[1]], 'mahalanobis')
        # print([mahalanobis(a, beta, np.linalg.inv(sigma)) for a, beta in zip(X[y == self.classes_[0]], X[y == self.classes_[1]])])

        return mahalanobis(np.mean(X[y == self.classes_[0]], axis=0),
                           np.mean(X[y == self.classes_[1]], axis=0),
                           self.pinv_sigma)


class MahalanobisBE(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, sigma=None, inv_sigma=None, sigma_method='full2', shrinkage=None, random_state=None, verbose=0):

        self.sigma = sigma
        self.inv_sigma = inv_sigma
        self.sigma_method = sigma_method
        self.shrinkage = shrinkage

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        # X = X[:, :3]

        # sigma = (np.cov(X[y == self.classes_[0]], rowvar=0) + np.cov(X[y == self.classes_[1]], rowvar=0))/2

        if self.sigma is None:
            self.sigma = _class_cov(X, y, shrinkage=self.shrinkage)
        if self.inv_sigma is None:
            self.inv_sigma = np.linalg.inv(self.sigma)

        return np.mean([mahalanobis(p[0], p[1], self.inv_sigma) for p in
                        product(X[y == self.classes_[0]], X[y == self.classes_[1]])])


class MahalanobisWI(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, sigma=None, inv_sigma=None, sigma_method='full2', shrinkage=None, random_state=None, verbose=0):

        self.sigma = sigma
        self.inv_sigma = inv_sigma
        self.sigma_method = sigma_method
        self.shrinkage = shrinkage

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        if self.sigma is None:
            self.sigma = _class_cov(X, y, shrinkage=self.shrinkage)
        if self.inv_sigma is None:
            self.inv_sigma = np.linalg.inv(self.sigma)

        return np.mean([mahalanobis(X[c[0]], X[c[1]], self.inv_sigma) for c in combinations(range(X.shape[0]), 2)])


class MahalanobisWC(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, sigma=None, inv_sigma=None, sigma_method='full2', shrinkage=None, random_state=None, verbose=0):

        self.sigma = sigma
        self.sigma_method = sigma_method
        self.inv_sigma = inv_sigma
        self.shrinkage = shrinkage

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        X1 = X[y == self.classes_[0]]
        X2 = X[y == self.classes_[1]]

        if self.sigma is None:
            self.sigma = _class_cov(X, y, shrinkage=self.shrinkage)
        if self.inv_sigma is None:
            self.inv_sigma = np.linalg.inv(self.sigma)

        wc = (np.mean([mahalanobis(X1[c[0]], X1[c[1]], self.inv_sigma) for c in combinations(range(X1.shape[0]), 2)]) +
              np.mean([mahalanobis(X2[c[0]], X2[c[1]], self.inv_sigma) for c in combinations(range(X2.shape[0]), 2)])) / 2

        return np.mean([mahalanobis(p[0], p[1], self.inv_sigma) for p in product(X1, X2)]) - wc

class MahalanobisWC2(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, sigma=None, inv_sigma=None, shrinkage=None, random_state=None, verbose=0):

        self.sigma = sigma
        self.inv_sigma = inv_sigma
        self.shrinkage = shrinkage

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        return self

    def predict(self, X, y):

        X = np.array(X)
        self.classes_ = np.unique(y)

        X1 = X[y == self.classes_[0]]
        X2 = X[y == self.classes_[1]]

        if self.sigma is None:
            self.sigma = _class_cov(X, y, shrinkage=self.shrinkage)
        if self.inv_sigma is None:
            self.inv_sigma = np.linalg.inv(self.sigma)

        wc = (np.mean([mahalanobis(X1[c[0]], X1[c[1]], self.inv_sigma) for c in combinations(range(X1.shape[0]), 2)]) +
              np.mean([mahalanobis(X2[c[0]], X2[c[1]], self.inv_sigma) for c in combinations(range(X2.shape[0]), 2)])) / 2

        return mahalanobis(np.mean(X1, axis=0), np.mean(X2, axis=0), self.inv_sigma) - wc



# class LDt(BaseEstimator):
#
#     _estimator_type = 'distance'
#
#     def __init__(self, sigma=None, random_state=None, verbose=0):
#
#         self.random_state = random_state
#         self.verbose = verbose
#         self.sigma = sigma
#
#     def fit(self, X, y):
#
#         X = np.array(X)
#         y = np.array(y)
#         # Xa = np.array(X).T
#         # Ya = np.tile(np.array(y), (X.shape[1], 1))
#
#         self.classes_ = np.unique(y)
#
#         # n_samples_class = np.unique(y, return_counts=True)[1]
#         # self.alpha = np.vstack((np.ones((n_samples_class[0], 1)), -np.ones((n_samples_class[1], 1))))
#         self.alpha = np.array([[1], [-1]])
#
#         # eBa = np.linalg.inv(Xa.T @ Xa) @ Xa.T @ Ya
#         # eEa = Ya - Xa @ eBa
#
#         # compute covariance matrix
#         # t, n = eEa.shape
#         # meanx = np.mean(eEa, axis=0)
#         # x = eEa - np.tile(meanx, (t, 1))
#         # sample = x.T @ x / t
#         # prior = np.diag(np.diag(sample))
#         # d = np.linalg.norm(sample - prior, ord='fro')**2 / n
#         # z = x**2 # z==y in Kriegeskorte toolbox
#         # r2 = 1/n/t**2*np.sum(z.T @ z) - 1/n/t*np.sum(sample**2)
#         # shrinkage = np.max((0, np.min((1, r2 / d))))
#         # self.sigma = shrinkage * prior + (1 - shrinkage) * sample
#
#         if self.sigma is None:
#             self.sigma = (np.cov(X[y == self.classes_[0]], rowvar=0) + np.cov(X[y == self.classes_[1]], rowvar=0))/2
#
#         invsigma = np.linalg.inv(self.sigma)
#         # self.was = self.alpha.T @ eBa @ invsigma
#         self.dist_train = np.mean(X[y == self.classes_[0]] - X[y == self.classes_[1]], axis=0)
#         self.was = self.dist_train @ invsigma
#
#         return self
    #
    # def predict(self, X, y):
    #
    #     X = np.array(X)
    #     y = np.array(y)
    #     # Xb = np.array(X).T
    #     # Yb = np.tile(np.array(y), (X.shape[1], 1))
    #     #
    #     # invXTXb = np.linalg.inv(Xb.T @ Xb)
    #     #
    #     # yb_was = Yb @ self.was.T
    #     # ebb_was = invXTXb @ Xb.T @ yb_was
    #     # eeb_was = yb_was - Xb @ ebb_was
    #     # nDFb = yb_was.shape[0] - Xb.shape[1]
    #     # esb_was = np.squeeze(np.diag(np.atleast_1d(eeb_was.T @ eeb_was))) / nDFb
    #     # C_new = self.alpha[:np.min((ebb_was.shape[0], self.alpha.shape[0])), :]
    #     # ctb_was2 = np.squeeze(np.diag(C_new.T @ ebb_was))
    #     #
    #     # se_ctb_was2 = np.sqrt(esb_was * np.diag(C_new.T @ invXTXb @ C_new))
    #     # ts = np.squeeze(ctb_was2 / se_ctb_was2)
    #
    #     dist_test = np.mean(X[y == self.classes_[0]] - X[y == self.classes_[1]], axis=0)
    #     # sigma_test = (np.cov(X[y == self.classes_[0]], rowvar=0) + np.cov(X[y == self.classes_[1]], rowvar=0))/2
    #
    #     # sigma2_error = self.was @ sigma_test @ self.was.T
    #     # sigma2_error = self.was @ (sigma_test * self.was.T)
    #     sigma2_error = max(self.was @ self.sigma @ self.was, 1e-5)
    #     SE_LDC = np.sqrt(sigma2_error)
    #
    #     return dist_test @ self.dist_train / SE_LDC

class LDt(BaseEstimator):

#  Alex Walther:
#  the ldt is the just the crossvalidated mahalanobis distance (i.e. a linear contrast) which has
#  been divisively normalized by the standard error (se). to compute se, we need sigma first. in
#  matlab speak, sigma would be
#
#  sigma=sum(w*Sw.*w,2);
#
#  Sw is the channel-by-channel variance-covariance matrix.

    _estimator_type = 'distance'

    def __init__(self, sigma=None, fractional_sigma=None, sigma_method='full2', inv_sigma=None, shrinkage=None,
                 random_state=None, verbose=0):

        self.sigma = sigma
        self.sigma_method = sigma_method
        self.fractional_sigma = fractional_sigma
        self.inv_sigma = inv_sigma
        self.shrinkage = shrinkage

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        if self.sigma is None:
            # self.sigma = (np.cov(X[y == self.classes_[0]], rowvar=0) + np.cov(X[y == self.classes_[1]], rowvar=0))/2
            self.sigma = _class_cov(X, y, shrinkage=self.shrinkage)
        if self.fractional_sigma is None:
            self.fractional_sigma = fractional_matrix_power(self.sigma, -0.5)
        if self.inv_sigma is None:
            self.inv_sigma = np.linalg.inv(self.sigma)

        X = X @ self.fractional_sigma

        self.dist_train = np.mean(X[y == self.classes_[0]] - X[y == self.classes_[1]], axis=0)

        self.w = self.dist_train @ self.inv_sigma

        return self

    def predict(self, X, y):

        X = np.array(X)
        y = np.array(y)

        X = X @ self.fractional_sigma

        dist_test = np.mean(X[y == self.classes_[0]] - X[y == self.classes_[1]], axis=0)

        sigma_small = self.w @ self.sigma @ self.w

        # SE_LDC = np.sqrt(max(1e-5, self.w @ self.sigma @ self.w))
        SE_LDC = np.sqrt(self.w * (sigma_small**2) @ self.w.T)

        return (self.dist_train@self.inv_sigma@dist_test) / SE_LDC


class PyRiemann(BaseEstimator):

    _estimator_type = 'classifier'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)


class LDA(LinearDiscriminantAnalysis):
    """Linear Discriminant Analysis

    A classifier with a linear decision boundary, generated by fitting class
    conditional densities to the data and using Bayes' rule.

    The model fits a Gaussian density to each class, assuming that all classes
    share the same covariance matrix.

    The fitted model can also be used to reduce the dimensionality of the input
    by projecting it to the most discriminative directions.

    .. versionadded:: 0.17
       *LinearDiscriminantAnalysis*.

    .. versionchanged:: 0.17
       Deprecated :class:`lda.LDA` have been moved to *LinearDiscriminantAnalysis*.

    Parameters
    ----------
    solver : string, optional
        Solver to use, possible values:
          - 'svd': Singular value decomposition (default). Does not compute the
                covariance matrix, therefore this solver is recommended for
                data with a large number of features.
          - 'lsqr': Least squares solution, can be combined with shrinkage.
          - 'eigen': Eigenvalue decomposition, can be combined with shrinkage.

    shrinkage : string or float, optional
        Shrinkage parameter, possible values:
          - None: no shrinkage (default).
          - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
          - float between 0 and 1: fixed shrinkage parameter.

        Note that shrinkage works only with 'lsqr' and 'eigen' solvers.

    priors : array, optional, shape (n_classes,)
        Class priors.

    n_components : int, optional
        Number of components (< n_classes - 1) for dimensionality reduction.

    store_covariance : bool, optional
        Additionally compute class covariance matrix (default False).

        .. versionadded:: 0.17

    tol : float, optional
        Threshold used for rank estimation in SVD solver.

        .. versionadded:: 0.17

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_classes, n_features)
        Weight vector(s).

    intercept_ : array, shape (n_features,)
        Intercept term.

    covariance_ : array-like, shape (n_features, n_features)
        Covariance matrix (shared by all classes).

    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.
        If ``n_components`` is not set then all components are stored and the
        sum of explained variances is equal to 1.0. Only available when eigen
        solver is used.

    means_ : array-like, shape (n_classes, n_features)
        Class means.

    priors_ : array-like, shape (n_classes,)
        Class priors (sum to 1).

    scalings_ : array-like, shape (rank, n_classes - 1)
        Scaling of the features in the space spanned by the class centroids.

    xbar_ : array-like, shape (n_features,)
        Overall mean.

    classes_ : array-like, shape (n_classes,)
        Unique class labels.

    See also
    --------
    sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis: Quadratic
        Discriminant Analysis

    Notes
    -----
    The default solver is 'svd'. It can perform both classification and
    transform, and it does not rely on the calculation of the covariance
    matrix. This can be an advantage in situations where the number of features
    is large. However, the 'svd' solver cannot be used with shrinkage.

    The 'lsqr' solver is an efficient algorithm that only works for
    classification. It supports shrinkage.

    The 'eigen' solver is based on the optimization of the between class
    scatter to within class scatter ratio. It can be used for both
    classification and transform, and it supports shrinkage. However, the
    'eigen' solver needs to compute the covariance matrix, so it might not be
    suitable for situations with a high number of features.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> clf = LinearDiscriminantAnalysis()
    >>> clf.fit(X, y)
    LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                  solver='svd', store_covariance=False, tol=0.0001)
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """
    def __init__(self, solver='lsqr', shrinkage=None, priors=None, n_components=None,
                 tol=1e-4,
                 sigma=None, sigma_method='full2'):

        super().__init__(solver=solver, shrinkage=shrinkage, priors=priors,
                         n_components=n_components, tol=tol)

        self.sigma = sigma
        self.sigma_method = sigma_method

    def _solve_lsqr(self, X, y, shrinkage):
        """Least squares solver.

        The least squares solver computes a straightforward solution of the
        optimal decision rule based directly on the discriminant functions. It
        can only be used for classification (with optional shrinkage), because
        estimation of eigenvectors is not performed. Therefore, dimensionality
        reduction with the transform is not supported.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,) or (n_samples, n_classes)
            Target values.

        shrinkage : string or float, optional
            Shrinkage parameter, possible values:
              - None: no shrinkage (default).
              - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
              - float between 0 and 1: fixed shrinkage parameter.

        Notes
        -----
        This solver is based on [1]_, section 2.6.2, pp. 39-41.

        References
        ----------
        .. [1] R. O. Duda, P. E. Hart, D. G. Stork. Pattern Classification
           (Second Edition). John Wiley & Sons, Inc., New York, 2001. ISBN
           0-471-05669-3.
        """
        self.means_ = _class_means(X, y)
        if self.sigma is not None:
            self.covariance_ = self.sigma
        else:
            self.covariance_ = _class_cov(X, y, self.priors_, shrinkage)
        self.coef_ = linalg.lstsq(self.covariance_, self.means_.T)[0].T
        self.intercept_ = (-0.5 * np.diag(np.dot(self.means_, self.coef_.T))
                           + np.log(self.priors_))