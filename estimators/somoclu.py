from sklearn.base import BaseEstimator
# from decereb.estimators.somoclu_train import Somoclu
from somoclu.train import Somoclu

import numpy as np


class SOM(BaseEstimator):

    _estimator_type = "classifier"

    def __init__(self, n_columns=10, n_rows=10, initialcodebook=None, kerneltype=0, maptype="planar",
                 gridtype="rectangular", compactsupport=False, neighborhood="gaussian", epochs=10,
                 radius0=0, radiusN=1, radiuscooling="linear", scale0=0.1, scaleN=0.01, scalecooling="linear"):

        self.som = Somoclu(n_columns, n_rows, initialcodebook, kerneltype, maptype, gridtype, compactsupport,
                           neighborhood, epochs, radius0, radiusN, radiuscooling, scale0, scaleN, scalecooling)

    def fit(self, X, y=None):

        self.som.train(X, y)

        return self

    def predict(self, X):

        # print([self.classes_[self.km.predict(x)] for x in X])

        return np.array([self.som.predict(x) for x in X])


