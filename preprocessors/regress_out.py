import numpy as np
from statsmodels.formula.api import ols

class RegressOut(object):

    def __init__(self, covariates=None):

        super().__init__()

        self.covariates = covariates
        # self.custom_training_ind = None
        self.Xcor_ = None

    def fit(self, X, y=None):

        X = np.array(X)

        self.Xcor_ = X.copy()
        # covariates_ = dict()
        # for covname, val in self.covariates.items():
        #     covariates_.update(covname=val[self.custom_training_ind])

        if len(self.covariates):
            for i in range(X.shape[1]):
                self.Xcor_[:, i] = ols('y ~ %s' % '+'.join(self.covariates.keys()), {**self.covariates, **dict(y=X[:, i])}).fit().resid.values + np.mean(X[:, i])

        return self

    def transform(self, X):

        return self.Xcor_