from abc import ABCMeta

import numpy as np
from sklearn.ensemble.base import BaseEnsemble
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
from sklearn.svm import LinearSVC
from sklearn.base import clone
from warnings import warn
from sklearn.cross_validation import LeaveOneOut


class RoiEnsemble(six.with_metaclass(ABCMeta, BaseEnsemble)):

    def __init__(self, verbose=0, n_jobs=1, random_state=None, base_estimator=LinearSVC(), base_estimator_args=None,
                 vote_graded=False):

        if base_estimator_args is None:
            base_estimator_args = dict()

        self.base_estimator = base_estimator
        self.base_estimator.set_params(**base_estimator_args)
        self.random_state = random_state
        self.set_random_state()
        #
        super().__init__(self.base_estimator, estimator_params=tuple(base_estimator_args.keys()))

        # for k, v in base_estimator_args.items():
        #     self.__setattr__(k, v)

        self.vote_graded = vote_graded

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

        if self.base_estimator._estimator_type == 'classifier':
            self.classes_ = np.unique(y)

        self.set_random_state()

        estimators = dict()
        for roi_id, x in X.items():
            estimator = clone(self.base_estimator)
            estimator.roi_id = roi_id
            if self.base_estimator._estimator_type == 'searchlight_ensemble':
                estimator.set_params(process_mask_img=x[1])
            estimators[roi_id] = estimator

        if self.vote_graded:
            y_pred = {k: np.full(len(y), np.nan) for k in X.keys()}
            for f, (train_index, test_index) in enumerate(LeaveOneOut(len(y))):
                y_train = [y[i] for i in train_index]

                if self.base_estimator._estimator_type == 'searchlight_ensemble':
                    estimators_fit = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                                     delayed(_parallel_build_estimator)(e, [X[roi_id][0][i] for i in train_index], y_train) for roi_id, e in estimators.items())
                    estimators_fit = {e.roi_id: e for e in estimators_fit}
                    y_pred_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                             delayed(_vote)(e, [X[roi_id][0][i] for i in test_index], False) for roi_id, e in estimators_fit.items())
                else:
                    estimators_fit = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                                     delayed(_parallel_build_estimator)(e, [X[roi_id][i] for i in train_index], y_train) for roi_id, e in estimators.items())
                    estimators_fit = {e.roi_id: e for e in estimators_fit}
                    y_pred_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                             delayed(_vote)(e, [X[roi_id][i] for i in test_index], False) for roi_id, e in estimators_fit.items())
                for i, roi_id in enumerate(X.keys()):
                    y_pred[roi_id][test_index] = y_pred_[i]

            self.vote_weighting = [np.mean(v == np.array(y)) for v in y_pred.values()]
            if not np.any(self.vote_weighting):
                self.vote_weighting = 1e-10 * np.ones(len(self.vote_weighting))
        else:
            self.vote_weighting = np.ones(len(X.keys())) / len(X.keys())




        if self.base_estimator._estimator_type == 'searchlight_ensemble':
            estimators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                         delayed(_parallel_build_estimator)(e, X[roi_id][0], y) for roi_id, e in estimators.items())
        else:
            estimators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                         delayed(_parallel_build_estimator)(e, X[roi_id], y) for roi_id, e in estimators.items())

        self.estimators_ = {e.roi_id: e for e in estimators}

        return self

    def predict_(self, X, probability=False):
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

        if self.base_estimator._estimator_type == 'searchlight_ensemble':
            self.votes = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                         delayed(_vote)(e, X[roi_id][0], probability) for roi_id, e in self.estimators_.items())
        else:
            self.votes = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                         delayed(_vote)(e, X[roi_id], probability) for roi_id, e in self.estimators_.items())

        self.votes_pooled = np.array(self.votes).swapaxes(0, 1).dot(self.vote_weighting) / sum(self.vote_weighting)

    def predict(self, X):

        self.predict_(X, False)

        if self.base_estimator._estimator_type == 'regressor':
            self.predictions = self.votes_pooled
        else:
            # self.predictions = self.classes_[(np.sign(self.votes_pooled) / 2. + 0.5).astype(int)]
            self.predictions = np.round(self.votes_pooled)

        return self.predictions

    def predict_graded(self, X):

        self.predict_(X)
        self.predictions = self.votes_pooled

        return self.predictions

    def predict_proba(self, X):

        self.predict_(X, probability=True)

        proba = 0.5 * np.ones((len(self.votes_pooled), 2))
        proba = proba + np.vstack((-self.votes_pooled, self.votes_pooled)).swapaxes(0, 1)

        return proba



def _vote(estimator, X, probability):
    """Private function used to compute a single vote in parallel."""

    if probability:
        if hasattr(estimator, 'predict_proba'):
            # vote = [-v[np.argmax(v)] * ((-1)**np.argmax(v)) for v in estimator.predict_proba(X)]
            vote = estimator.predict_proba(X)[:, 1] - 0.5
        # elif hasattr(estimator, 'decision_function'):
        #     vote = estimator.decision_function(X)
        else:
            warn('Estimator must implement predict_proba if probability=True. Using predict() instead.')
            vote = estimator.predict(X)
    else:
        vote = estimator.predict(X)

    return vote

def _parallel_build_estimator(estimator, X, y):
    """Private function used to fit a single estimator in parallel."""

    estimator.fit(X, y)

    return estimator