from sklearn.base import BaseEstimator
import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean

class CvPearson(BaseEstimator):

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
        self.denom = np.sqrt(np.cov(self.A1) * np.cov(self.B1))

        return self

    def predict(self, X, y):

        X = np.array(X)

        A2 = np.mean(X[y == self.classes_[0]], axis=0)
        B2 = np.mean(X[y == self.classes_[1]], axis=0)

        r = (np.cov(self.A1, B2)[0, 1] + np.cov(self.B1, A2)[0, 1]) / (2 * self.denom)

        return r

class CvPearson2(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0, minval=None):

        self.random_state = random_state
        self.verbose = verbose
        self.minval = minval

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

        denom = np.sqrt(np.cov(self.A1, A2)[0, 1] * np.cov(self.B1, B2)[0, 1])
        if self.minval is not None and denom < self.minval:
            denom = self.minval

        r = (np.cov(self.A1, B2)[0, 1] + np.cov(self.B1, A2)[0, 1]) / (2 * denom)

        return r


class CvPearsonClassifier(BaseEstimator):

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
        self.denom = np.sqrt(np.cov(self.A1) * np.cov(self.B1))

        return self

    def predict(self, X):

        X = np.array(X)

        predictions = [self.classes_[int(pearsonr(self.B1, x)[0] > pearsonr(self.A1, x)[0])] for x in X]

        return predictions


class CvEuclidian(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        self.dist_train = np.mean(X[y == self.classes_[0]] - X[y == self.classes_[1]], axis=0)

        return self

    def predict(self, X, y):

        X = np.array(X)
        y = np.array(y)

        dist_test = np.mean(X[y == self.classes_[0]] - X[y == self.classes_[1]], axis=0)

        return np.dot(self.dist_train, dist_test)

class CvEuclidianClassifier(BaseEstimator):

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

        predictions = [self.classes_[int(euclidean(self.A1, x) > euclidean(self.B1, x))] for x in X]

        return predictions


class CvMahalanobis(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, sigma=None, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose
        self.sigma = sigma

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        if self.sigma is None:
            self.sigma = (np.cov(X[y == self.classes_[0]], rowvar=0) + np.cov(X[y == self.classes_[1]], rowvar=0))/2

        self.dist_train = np.mean(X[y == self.classes_[0]] - X[y == self.classes_[1]], axis=0)

        return self

    def predict(self, X, y):

        X = np.array(X)
        y = np.array(y)

        dist_test = np.mean(X[y == self.classes_[0]] - X[y == self.classes_[1]], axis=0)

        return self.dist_train@self.sigma@dist_test

class LDt(BaseEstimator):

    _estimator_type = 'distance'

    def __init__(self, sigma=None, random_state=None, verbose=0):

        self.random_state = random_state
        self.verbose = verbose
        self.sigma = sigma

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y)
        # Xa = np.array(X).T
        # Ya = np.tile(np.array(y), (X.shape[1], 1))

        self.classes_ = np.unique(y)

        # n_samples_class = np.unique(y, return_counts=True)[1]
        # self.C = np.vstack((np.ones((n_samples_class[0], 1)), -np.ones((n_samples_class[1], 1))))
        self.C = np.array([[1], [-1]])

        # eBa = np.linalg.inv(Xa.T @ Xa) @ Xa.T @ Ya
        # eEa = Ya - Xa @ eBa

        # compute covariance matrix
        # t, n = eEa.shape
        # meanx = np.mean(eEa, axis=0)
        # x = eEa - np.tile(meanx, (t, 1))
        # sample = x.T @ x / t
        # prior = np.diag(np.diag(sample))
        # d = np.linalg.norm(sample - prior, ord='fro')**2 / n
        # z = x**2 # z==y in Kriegeskorte toolbox
        # r2 = 1/n/t**2*np.sum(z.T @ z) - 1/n/t*np.sum(sample**2)
        # shrinkage = np.max((0, np.min((1, r2 / d))))
        # self.sigma = shrinkage * prior + (1 - shrinkage) * sample

        if self.sigma is None:
            self.sigma = (np.cov(X[y == self.classes_[0]], rowvar=0) + np.cov(X[y == self.classes_[1]], rowvar=0))/2

        invsigma = np.linalg.inv(self.sigma)
        # self.was = self.C.T @ eBa @ invsigma
        self.dist_train = np.mean(X[y == self.classes_[0]] - X[y == self.classes_[1]], axis=0)
        self.was = self.dist_train @ invsigma

        return self

    def predict(self, X, y):

        X = np.array(X)
        y = np.array(y)
        # Xb = np.array(X).T
        # Yb = np.tile(np.array(y), (X.shape[1], 1))
        #
        # invXTXb = np.linalg.inv(Xb.T @ Xb)
        #
        # yb_was = Yb @ self.was.T
        # ebb_was = invXTXb @ Xb.T @ yb_was
        # eeb_was = yb_was - Xb @ ebb_was
        # nDFb = yb_was.shape[0] - Xb.shape[1]
        # esb_was = np.squeeze(np.diag(np.atleast_1d(eeb_was.T @ eeb_was))) / nDFb
        # C_new = self.C[:np.min((ebb_was.shape[0], self.C.shape[0])), :]
        # ctb_was2 = np.squeeze(np.diag(C_new.T @ ebb_was))
        #
        # se_ctb_was2 = np.sqrt(esb_was * np.diag(C_new.T @ invXTXb @ C_new))
        # ts = np.squeeze(ctb_was2 / se_ctb_was2)

        dist_test = np.mean(X[y == self.classes_[0]] - X[y == self.classes_[1]], axis=0)
        # sigma_test = (np.cov(X[y == self.classes_[0]], rowvar=0) + np.cov(X[y == self.classes_[1]], rowvar=0))/2

        # sigma2_error = self.was @ sigma_test @ self.was.T
        # sigma2_error = self.was @ (sigma_test * self.was.T)
        sigma2_error = self.was @ self.sigma @ self.was.T
        SE_LDC = np.sqrt(sigma2_error)

        return dist_test @ self.dist_train / SE_LDC