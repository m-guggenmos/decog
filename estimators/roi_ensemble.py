from abc import ABCMeta

import numpy as np
from sklearn.ensemble.base import BaseEstimator, BaseEnsemble
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
from sklearn.svm import LinearSVC
from inspect import signature
from scipy.stats import mode
from sklearn.base import clone
from warnings import warn
from sklearn.externals.six import iteritems
from sklearn.model_selection import LeaveOneOut
import time


class MetaClassifier(BaseEstimator):

    def __init__(self):
        self.estimator_list = None
        self.random_state = None

    def set_random_state(self, random_state):
        self.random_state = random_state
        for estimator in self.estimator_list:
            if hasattr(estimator[1], 'random_state'):
                estimator[1].set_params(random_state=random_state)


class MultiModalRoiEnsembleBayes(MetaClassifier):

    def __init__(self, base_estimators=None, verbose=0, n_jobs=1, random_state=None):

        super().__init__()

        self.base_estimators = base_estimators
        self.set_random_state(random_state)

        self.n_jobs = n_jobs
        self.verbose = verbose

        self.estimators_ = []
        self.classes_ = None
        self.priors = dict()

    @property
    def base_estimators(self):
        return self._base_estimators

    @base_estimators.setter
    def base_estimators(self, value):
        assert isinstance(value, dict), 'base_estimators argument for MultiModalRoiEnsembleBayes ' \
                                        'has to be a dictionary'
        self._base_estimators = value
        self.estimator_list = []
        for k, v in self._base_estimators.items():
            self.estimator_list.append((k, v))

    # def set_base_estimators(self, base_estimators):
    #     assert isinstance(base_estimators, dict), 'base_estimators argument for MultiModalRoiEnsembleBayes ' \
    #                                     'has to be a dictionary'
    #     self.estimator_list = []
    #     for k, v in base_estimators.items():
    #         setattr(self, 'estimator_%s' % k, v)
    #         self.estimator_list.append((k, v))

    # def set_random_state(self):
    #     for estimator in self.catalogue:
    #         if hasattr(getattr(self, estimator), 'random_state'):
    #             getattr(self, estimator).set_params(random_state=self.random_state)

    def fit(self, X, y):
        """Fit estimators from the training set (X, y).

        Returns
        -------
        self : object
            Returns self.
        """

        if not isinstance(X, dict):
            raise ValueError("X has to be a dict")

        self.classes_ = np.unique(y)

        estimators = dict()
        for modality, Xm in X.items():
            for roi_id, x in Xm.items():
                estimator = clone(self.base_estimators[modality])
                estimator.id = (modality, roi_id)
                estimators[estimator.id] = estimator

        y_pred = {k: np.full(len(y), np.nan) for k in estimators.keys()}
        t0 = time.time()
        print('Start [1]')
        for f, (train_index, test_index) in enumerate(LeaveOneOut()):
            y_train = [y[i] for i in train_index]

            estimators_fit = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(delayed(_parallel_build_estimator)(
                e, np.array([X[id[0]][id[1]][i] for i in train_index]), y_train) for id, e in estimators.items())
            estimators_fit = {e.id: e for e in estimators_fit}
            # for roi_id, e in estimators_fit.items():
            #     e.predict([X[roi_id][i] for i in test_index])
            y_pred_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                     delayed(_vote)(e, [X[id[0]][id[1]][i] for i in test_index], False) for id, e in estimators_fit.items())
            for i, id in enumerate(estimators.keys()):
                y_pred[id][test_index] = y_pred_[i]
        print('[1] Elapsed time: %.2f secs' % (time.time() - t0))

        for i, id in enumerate(estimators.keys()):
            self.priors[(self.classes_[0], self.classes_[0])] = np.mean(y_pred[id][y == self.classes_[0]] == self.classes_[0])
            self.priors[(self.classes_[1], self.classes_[0])] = 1 - self.priors[(self.classes_[0], self.classes_[0])]
            self.priors[(self.classes_[1], self.classes_[1])] = np.mean(y_pred[id][y == self.classes_[1]] == self.classes_[1])
            self.priors[(self.classes_[0], self.classes_[1])] = 1 - self.priors[(self.classes_[1], self.classes_[1])]

        t0 = time.time()
        estimators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                     delayed(_parallel_build_estimator)(e, X[id[0]][id[1]], y) for id, e in estimators.items())
        print('[2] Elapsed time: %.2f secs' % (time.time() - t0))

        self.estimators_ = {e.id: e for e in estimators}

        return self

    def predict(self, X, probability=False):
        """Predict class for X.

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

        votes = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                     delayed(_vote)(e, X[id[0]][id[1]], probability) for id, e in self.estimators_.items())

        ns = list(list(X.values())[0].values())[0].shape[0]

        prior_hyp = {self.classes_[0]: 0.5*np.ones(ns), self.classes_[1]: 0.5*np.ones(ns)}

        for n in range(ns):
            for i in range(len(votes)):
                prior_hyp[self.classes_[1]][n] = (self.priors[(votes[i][n], self.classes_[1])] * prior_hyp[self.classes_[1]][n]) / \
                    (self.priors[(votes[i][n], self.classes_[1])] * prior_hyp[self.classes_[1]][n] + self.priors[(votes[i][n], self.classes_[0])] * prior_hyp[self.classes_[0]][n])
                prior_hyp[self.classes_[0]][n] = 1 - prior_hyp[self.classes_[1]][n]

        return np.array([self.classes_[np.argmax([prior_hyp[self.classes_[0]][n], prior_hyp[self.classes_[1]][n]])] for n in range(ns)])


    def get_params(self, deep=True):
        if not deep:
            return super(MultiModalRoiEnsembleBayes, self).get_params(deep=False)
        else:
            out = dict(self.estimator_list)
            for name, est in self.estimator_list:
                for key, value in iteritems(est.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            out.update(super(MultiModalRoiEnsembleBayes, self).get_params(deep=False))
            return out


class MultiModalBayes(MetaClassifier):

    def __init__(self, base_estimators=None, verbose=0, n_jobs=1, random_state=None):

        super().__init__()

        self.base_estimators = base_estimators
        self.set_random_state(random_state)

        self.n_jobs = n_jobs
        self.verbose = verbose

        self.estimators_ = []
        self.classes_ = None
        self.priors = dict()

    @property
    def base_estimators(self):
        return self._base_estimators

    @base_estimators.setter
    def base_estimators(self, value):
        assert isinstance(value, dict), 'base_estimators argument for MultiModalRoiEnsembleBayes ' \
                                        'has to be a dictionary'
        self._base_estimators = value
        self.estimator_list = []
        for k, v in self._base_estimators.items():
            self.estimator_list.append((k, v))

    # def set_base_estimators(self, base_estimators):
    #     assert isinstance(base_estimators, dict), 'base_estimators argument for MultiModalRoiEnsembleBayes ' \
    #                                     'has to be a dictionary'
    #     self.estimator_list = []
    #     for k, v in base_estimators.items():
    #         setattr(self, 'estimator_%s' % k, v)
    #         self.estimator_list.append((k, v))

    # def set_random_state(self):
    #     for estimator in self.catalogue:
    #         if hasattr(getattr(self, estimator), 'random_state'):
    #             getattr(self, estimator).set_params(random_state=self.random_state)

    def fit(self, X, y):
        """Fit estimators from the training set (X, y).

        Returns
        -------
        self : object
            Returns self.
        """

        if not isinstance(X, dict):
            raise ValueError("X has to be a dict")

        self.classes_ = np.unique(y)

        estimators = dict()
        for modality, Xm in X.items():
            estimator = clone(self.base_estimators[modality])
            estimator.id = modality
            estimators[modality] = estimator

        t0 = time.time()
        print('Start [1]')
        # y_pred_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
        #     delayed(_build_priors)(X, y, train_index, test_index, estimators, self.n_jobs, self.verbose) for train_index, test_index in LeaveOneOut())
        # y_pred_ = np.squeeze(np.array(y_pred_)).T
        # y_pred = dict()
        # for i, id in enumerate(estimators.keys()):
        #     y_pred[id] = y_pred_[i]

        y_pred = {k: np.full(len(y), np.nan) for k in estimators.keys()}
        for f, (train_index, test_index) in enumerate(LeaveOneOut()):
            y_train = [y[i] for i in train_index]

            estimators_fit = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(delayed(_parallel_build_estimator)(
                e, np.array([X[id][i] for i in train_index]), y_train) for id, e in estimators.items())
            estimators_fit = {e.id: e for e in estimators_fit}
            # for roi_id, e in estimators_fit.items():
            #     e.predict([X[roi_id][i] for i in test_index])
            y_pred_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                     delayed(_vote)(e, [X[id][i] for i in test_index], False) for id, e in estimators_fit.items())
            for i, id in enumerate(estimators.keys()):
                y_pred[id][test_index] = y_pred_[i]
        print('[1] Elapsed time: %.2f secs' % (time.time() - t0))

        for i, id in enumerate(estimators.keys()):
            self.priors[(self.classes_[0], self.classes_[0])] = np.mean(y_pred[id][y == self.classes_[0]] == self.classes_[0])
            self.priors[(self.classes_[1], self.classes_[0])] = 1 - self.priors[(self.classes_[0], self.classes_[0])]
            self.priors[(self.classes_[1], self.classes_[1])] = np.mean(y_pred[id][y == self.classes_[1]] == self.classes_[1])
            self.priors[(self.classes_[0], self.classes_[1])] = 1 - self.priors[(self.classes_[1], self.classes_[1])]

        estimators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                     delayed(_parallel_build_estimator)(e, X[id], y) for id, e in estimators.items())

        self.estimators_ = {e.id: e for e in estimators}

        return self

    def predict(self, X, probability=False):
        """Predict class for X.

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

        votes = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                     delayed(_vote)(e, X[id], probability) for id, e in self.estimators_.items())

        nsub = len(list(X.values())[0])

        prior_hyp = {self.classes_[0]: 0.5*np.ones(nsub), self.classes_[1]: 0.5*np.ones(nsub)}

        for s in range(nsub):
            for i in range(len(votes)):
                prior_hyp[self.classes_[1]][s] = (self.priors[(votes[i][s], self.classes_[1])] * prior_hyp[self.classes_[1]][s]) / (self.priors[(votes[i][s], self.classes_[1])] * prior_hyp[self.classes_[1]][s] + self.priors[(votes[i][s], self.classes_[0])] * prior_hyp[self.classes_[0]][s])
                prior_hyp[self.classes_[0]][s] = 1 - prior_hyp[self.classes_[1]][s]

        return np.array([self.classes_[np.argmax([prior_hyp[self.classes_[0]][s], prior_hyp[self.classes_[1]][s]])] for s in range(nsub)])


    def get_params(self, deep=True):
        if not deep:
            return super(MultiModalBayes, self).get_params(deep=False)
        else:
            out = dict(self.estimator_list)
            for name, est in self.estimator_list:
                for key, value in iteritems(est.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            out.update(super(MultiModalBayes, self).get_params(deep=False))
            return out



class RoiEnsembleBayes(BaseEstimator):

    def __init__(self, verbose=0, n_jobs=1, random_state=None, base_estimator=LinearSVC()):

        self.base_estimator = base_estimator
        self.random_state = random_state
        self.set_random_state()

        self.n_jobs = n_jobs
        self.verbose = verbose

        self.estimators_ = []
        self.classes_ = None
        self.priors = dict()

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

        y_pred = {k: np.full(len(y), np.nan) for k in X.keys()}
        for f, (train_index, test_index) in enumerate(LeaveOneOut()):
            y_train = [y[i] for i in train_index]

            estimators_fit = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                             delayed(_parallel_build_estimator)(e, np.array([X[roi_id][i] for i in train_index]), y_train) for roi_id, e in estimators.items())
            estimators_fit = {e.roi_id: e for e in estimators_fit}
            # for roi_id, e in estimators_fit.items():
            #     e.predict([X[roi_id][i] for i in test_index])
            y_pred_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                     delayed(_vote)(e, [X[roi_id][i] for i in test_index], False) for roi_id, e in estimators_fit.items())
            for i, roi_id in enumerate(X.keys()):
                y_pred[roi_id][test_index] = y_pred_[i]

        for i, roi_id in enumerate(X.keys()):
            self.priors[(self.classes_[0], self.classes_[0])] = np.mean(y_pred[roi_id][y == self.classes_[0]] == self.classes_[0])
            self.priors[(self.classes_[1], self.classes_[0])] = 1 - self.priors[(self.classes_[0], self.classes_[0])]
            self.priors[(self.classes_[1], self.classes_[1])] = np.mean(y_pred[roi_id][y == self.classes_[1]] == self.classes_[1])
            self.priors[(self.classes_[0], self.classes_[1])] = 1 - self.priors[(self.classes_[1], self.classes_[1])]

        estimators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                     delayed(_parallel_build_estimator)(e, X[roi_id], y) for roi_id, e in estimators.items())

        self.estimators_ = {e.roi_id: e for e in estimators}

        return self

    def predict(self, X, probability=False):
        """Predict class for X.

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

        votes = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                     delayed(_vote)(e, X[roi_id], probability) for roi_id, e in self.estimators_.items())

        ns = X[0].shape[0]

        prior_hyp = {self.classes_[0]: 0.5*np.ones(ns), self.classes_[1]: 0.5*np.ones(ns)}

        for n in range(ns):
            for i, roi_id in enumerate(X.keys()):
                prior_hyp[self.classes_[1]][n] = (self.priors[(votes[i][n], self.classes_[1])] * prior_hyp[self.classes_[1]][n]) / \
                    (self.priors[(votes[i][n], self.classes_[1])] * prior_hyp[self.classes_[1]][n] + self.priors[(votes[i][n], self.classes_[0])] * prior_hyp[self.classes_[0]][n])
                prior_hyp[self.classes_[0]][n] = 1 - prior_hyp[self.classes_[1]][n]

        return np.array([self.classes_[np.argmax([prior_hyp[self.classes_[0]][n], prior_hyp[self.classes_[1]][n]])] for n in range(ns)])



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
            for f, (train_index, test_index) in enumerate(LeaveOneOut()):
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

        if False in np.isfinite(self.votes):
            self.votes_pooled = [np.nan]
        else:
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
        if hasattr(estimator, 'classes_') and estimator.classes_ is not None:
            vote = estimator.predict(X)
        else:
            vote = np.nan

    return vote

def _parallel_build_estimator(estimator, X, y):
    """Private function used to fit a single estimator in parallel."""

    if isinstance(X, np.ndarray) and X.ndim == 2:
        estimator.fit(X, y)

    return estimator

def _build_priors(X, y, train_index, test_index, estimators, n_jobs, verbose):
    y_train = [y[i] for i in train_index]
    # n_jobs = 1

    estimators_fit = Parallel(n_jobs=n_jobs, verbose=verbose, backend="threading")(
        delayed(_parallel_build_estimator)(
            e, np.array([X[id][i] for i in train_index]), y_train) for id, e in
        estimators.items())
    estimators_fit = {e.id: e for e in estimators_fit}
    # for roi_id, e in estimators_fit.items():
    #     e.predict([X[roi_id][i] for i in test_index])
    y_pred_ = Parallel(n_jobs=n_jobs, verbose=verbose, backend="threading")(
        delayed(_vote)(e, [X[id][i] for i in test_index], False) for id, e in
        estimators_fit.items())
    # y_pred = dict()
    # for i, id in enumerate(estimators.keys()):
    #     y_pred[id] = y_pred_[i]
    return y_pred_