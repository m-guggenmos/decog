from abc import ABCMeta

import numpy as np
from sklearn.ensemble.base import BaseEnsemble
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
from sklearn.svm import LinearSVC
from inspect import signature
from scipy.stats import mode
from sklearn.base import clone

class RoiEnsemble(six.with_metaclass(ABCMeta, BaseEnsemble)):

    def __init__(self, verbose=0, n_jobs=1, random_state=None, base_estimator=LinearSVC(), base_estimator_args=None,
                 searchlight=False, regression=False, continuous=False):

        """

        Args:
            combine_rois (bool): whether data of rois should be combined or act as independent votes
            searchlight (bool): whether the classifier is searchlight-based

        """
        if base_estimator_args is None:
            base_estimator_args = dict()

        self.base_estimator = base_estimator
        self.base_estimator.set_params(**base_estimator_args)
        self.random_state = random_state
        self.set_random_state()
        #
        super(RoiEnsemble, self).__init__(self.base_estimator, estimator_params=tuple(base_estimator_args.keys()))

        # for k, v in base_estimator_args.items():
        #     self.__setattr__(k, v)

        self.searchlight = searchlight
        self.regression = regression
        self.continuous = continuous

        self.n_jobs = n_jobs
        self.verbose = verbose

        self.estimators_ = []
        self.classes_ = None

    def set_random_state(self):
        if hasattr(self.base_estimator, 'random_state'):
            self.base_estimator.set_params(random_state=self.random_state)

    def fit(self, X, y):
        """Fit estimators from the training set (X, y).

        Returns
        -------
        self : object
            Returns self.
        """

        if not isinstance(X, dict):
            raise ValueError("X has to be a dict")

        if not self.regression:
            self.classes_ = np.unique(y)

        self.set_random_state()

        estimators = dict()
        for roi_id, x in X.items():
            estimator = clone(self.base_estimator)
            estimator.roi_id = roi_id
            if self.searchlight:
                estimator.set_params(process_mask_img=x[1])
            estimators[roi_id] = estimator

        if self.searchlight:
            estimators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                         delayed(_parallel_build_estimator)(e, X[roi_id][0], y) for roi_id, e in estimators.items())
        else:
            estimators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                         delayed(_parallel_build_estimator)(e, X[roi_id], y) for roi_id, e in estimators.items())

        self.estimators_ = {e.roi_id: e for e in estimators}

        return self

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is a vote by the individual searchlights.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes.
        """

        # votes = []
        # for v in range(self.n_best):
        #     votes += [self.estimators_[v].predict(np.array([x.get_data()[self.best_spheres[v]] for x in X]))]

        if not isinstance(X, dict):
            raise ValueError("X has to be a dict")

        if self.searchlight:
            self.votes = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                         delayed(_vote)(e, X[roi_id][0], self.continuous) for roi_id, e in self.estimators_.items())
        else:
            self.votes = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                         delayed(_vote)(e, X[roi_id], self.continuous) for roi_id, e in self.estimators_.items())

        if self.regression:
            self.votes_pooled = np.mean(self.votes, axis=0)
            self.predictions = self.votes_pooled
        else:
            if self.continuous:
                if len(X) == 1:
                    vote = np.sign(self.votes[0]) / 2. + 0.5
                    self.votes_pooled = vote
                else:
                    self.votes_pooled = np.mean(self.votes, axis=0)
                    vote = np.sign(self.votes_pooled) / 2. + 0.5
                self.predictions = self.classes_[vote.astype(int)]
            else:
                self.predictions = mode(self.votes)[0][0]


        return self.predictions

    # def set_params(self, **params):
    #     super().set_params(**params)
    #     if 'random_state' in params:
    #         self.set_random_state(params['random_state'])


def _vote(estimator, X, continuous):
    """Private function used to compute a single vote in parallel."""

    if continuous:
        # vote = estimator.predict(X)
        if hasattr(estimator, 'decision_function'):
            vote = estimator.decision_function(X)
        elif hasattr(estimator, 'predict_proba'):
            vote = [-v[np.argmax(v)] * ((-1)**np.argmax(v)) for v in estimator.predict_proba(X)]
        else:
            AttributeError('Estimator must either implement decision_function or predict_proba if continuous=True.')
    else:
        vote = estimator.predict(X)

    return vote

def _parallel_build_estimator(estimator, X, y):
    """Private function used to fit a single estimator in parallel."""

    estimator.fit(X, y)

    return estimator