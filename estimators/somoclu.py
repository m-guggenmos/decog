from sklearn.base import BaseEstimator
from mgmvpa.estimators.somoclu import Somoclu

class SOM(BaseEstimator):

    def __init__(self, n_columns=10, n_rows=10):

        self.som = Somoclu(n_columns, n_rows)

    def fit(self, X, y=None):

        self.som.train(X, y)

        return self

    def predict(self, X):

        # print([self.classes_[self.km.predict(x)] for x in X])

        return [self.som.predict(x) for x in X]


