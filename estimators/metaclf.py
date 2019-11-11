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
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from itertools import product, islice
import importlib
from collections import OrderedDict

class MetaClassifier(BaseEstimator):

    def __init__(self):
        self.estimator_list = None
        self.random_state = None

    def set_random_state(self, random_state):
        self.random_state = random_state
        for estimator in self.estimator_list:
            if hasattr(estimator[1], 'random_state'):
                estimator[1].set_params(random_state=random_state)

class MultiModalMetaClassifier(MetaClassifier):

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

        for k, estimator in self.base_estimators.items():
            estimator.fit(X[k], y)


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

        n_sample = len(X[list(X.keys())[0]])
        n_estimators = len(self.base_estimators)
        pred = np.full((n_estimators, n_sample), np.nan)
        for i, (k, estimator) in enumerate(self.base_estimators.items()):
            pred[i] = estimator.predict(X[k])

        return np.round(pred.mean(axis=0))


class MultiModalProbabilisticMetaClassifier(MetaClassifier):

    def __init__(self, base_estimators=None, verbose=0, n_jobs=1, random_state=None,
                 weight_grid=None, weight_cv=None, weight_separate=False, weight_provided=None,
                 weight_fixed=None, weight_scoring='balanced_accuracy', discrete=False):

        super().__init__()

        self.base_estimators = base_estimators
        self.set_random_state(random_state)

        self.n_jobs = n_jobs
        self.verbose = verbose

        self.estimators_ = []
        self.classes_ = None
        self.weight_grid = weight_grid
        self.weight_cv = weight_cv
        self.weight_separate = weight_separate
        self.weighting = None
        self.weight_provided = weight_provided
        self.weight_fixed = weight_fixed
        self.weight_scoring = weight_scoring
        self.discrete = discrete
        self.priors = dict()

    @property
    def base_estimators(self):
        return self._base_estimators

    @base_estimators.setter
    def base_estimators(self, value):
        assert isinstance(value, dict), 'base_estimators argument for MultiModalProbabilisticMetaClassifier ' \
                                        'has to be a dictionary'
        self._base_estimators = value
        self.estimator_list = []
        for k, v in self._base_estimators.items():
            self.estimator_list.append((k, v))

    def fit(self, X, y):
        return self.fit_(X, y, weight_grid=self.weight_grid, weight_provided=self.weight_provided)

    def fit_(self, X, y, weight_grid=None, weight_provided=None):

        y = np.array(y)
        X = OrderedDict({m: np.array(x) for m, x in X.items()})

        if not isinstance(X, dict):
            raise ValueError("X has to be a dict")

        self.classes_ = np.unique(y)

        for k, estimator in self.base_estimators.items():
            valid = ~np.isnan(X[k][:, 0])
            estimator.fit(X[k][valid], y[valid])

        if weight_grid is not None or weight_provided is not None:
            best_combo = self._get_best_combo(X, y, weight_grid)
            self.weighting = OrderedDict({m: best_combo[i] for i, m in enumerate(X.keys())})
            if self.verbose:
                print('Weighting: %s' % self.weighting)

        return self

    # def _get_best_combo(self, X, y, weight_grid):
    #
    #     cv = StratifiedKFold(5) if self.weight_cv is None else self.weight_cv
    #     n_folds = cv.get_n_splits(X[list(X.keys())[0]], y)
    #     scoringfunc = getattr(importlib.import_module('sklearn.metrics'), '%s_score' % self.weight_scoring)
    #     estimator = clone(self)
    #
    #     if hasattr(self, 'weight_provided') and self.weight_provided is not None:
    #         combos = self.weight_provided
    #         n_combos = len(combos)
    #     elif self.weight_separate:
    #         grid0 = np.vstack((weight_grid, np.ones((len(X) - 1, len(weight_grid))))).T
    #         combos = np.vstack([np.roll(grid0, r, axis=1) for r in range(len(X))]).tolist()
    #         n_combos = len(combos)
    #         scores = np.full(n_combos, np.nan)
    #     else:
    #         combos = product(weight_grid, repeat=len(X))
    #         # uniq = np.unique([np.array(c)/np.sum(c) for c in combos], axis=0, return_index=True)[1]
    #         # combos = [combos[i] for i in uniq]
    #         n_combos = len(weight_grid) ** len(X)
    #         best_score = -1
    #
    #     modulus = max(1, int(10**len(str(n_combos)) / 100.))
    #     for i, combo in enumerate(combos):
    #         # if not i or not np.mod(i, modulus):
    #         #     print('\t[%s] Combo %g / %s %s' % (time.strftime("%d.%m %H:%M:%S"), i + 1, n_combos, combo))
    #         pred = [None] * n_folds
    #         y_true = [None] * n_folds
    #         for f, (train, test) in enumerate(cv.split(X[list(X.keys())[0]], y)):
    #             X_train = OrderedDict({m: x[train] for m, x in X.items()})
    #             X_test = OrderedDict({m: x[test] for m, x in X.items()})
    #             estimator.fit_(X_train, y[train])
    #             weighting = OrderedDict({m: combo[i] for i, m in enumerate(X.keys())})
    #             pred[f] = estimator.predict_(X_test, weighting=weighting).tolist()
    #             y_true[f] = y[test].tolist()
    #         score = scoringfunc(sum(y_true, []), sum(pred, []))
    #         print('\t[%s] Combo %g / %s %s %.2f' % (time.strftime("%d.%m %H:%M:%S"), i + 1, n_combos, combo, 100*score))
    #         if self.weight_separate:
    #             scores[i] = score
    #         elif score > best_score:
    #             best_combo = combo
    #             best_score = score
    #     if self.weight_separate:
    #         best_combo = [weight_grid[np.argmax(scores[i*len(weight_grid):(i+1)*len(weight_grid)])] for i in range(len(X))]
    #
    #     print('\tBest combo: %s' % (best_combo, ))
    #
    #     return best_combo


    def _get_best_combo(self, X, y, weight_grid):

        cv = StratifiedKFold(5) if self.weight_cv is None else self.weight_cv
        n_folds = cv.get_n_splits(X[list(X.keys())[0]], y)
        scoringfunc = getattr(importlib.import_module('sklearn.metrics'), '%s_score' % self.weight_scoring)
        estimator = clone(self)

        if self.weight_provided is not None:
            combos = self.weight_provided
            n_combos = len(combos)
        elif self.weight_separate:
            grid0 = np.vstack((weight_grid, np.ones((len(X) - 1, len(weight_grid))))).T
            combos = np.vstack([np.roll(grid0, r, axis=1) for r in range(len(X))]).tolist()
            n_combos = len(combos)
        else:
            combos = product(weight_grid, repeat=len(X))
            # uniq = np.unique([np.array(c)/np.sum(c) for c in combos], axis=0, return_index=True)[1]
            # combos = [combos[i] for i in uniq]
            # n_combos = len(combos)
            n_combos = len(weight_grid) ** len(X)

        modulus = max(1, int(10 ** (len(str(n_combos))-1) / 100.))
        scores = Parallel(n_jobs=self.n_jobs)(delayed(self._combo)(i, clone(estimator), cv, X, y, combo, scoringfunc, n_folds, n_combos, modulus) for i, combo in enumerate(combos))
        if self.weight_provided is not None:
            best_combo = combos[np.argmax(scores)]
        elif self.weight_separate:
            best_combo = [weight_grid[np.argmax(np.array(scores)[i * len(weight_grid):(i + 1) * len(weight_grid)])] for i in range(len(X))]
        else:
            best_combo = next(islice(product(weight_grid, repeat=len(X)), int(np.argmax(scores)), None))

        print('\tBest combo: %s' % (best_combo, ))

        return best_combo

    def _combo(self, i, estimator, cv, X, y, combo, scoringfunc, n_folds, n_combos, modulus):
        # if not i or not np.mod(i, modulus):
        #     print('\t[%s] Combo %g / %s %s' % (time.strftime("%d.%m %H:%M:%S"), i + 1, n_combos, combo))
        pred = [None] * n_folds
        y_true = [None] * n_folds
        for f, (train, test) in enumerate(cv.split(X[list(X.keys())[0]], y)):
            X_train = OrderedDict({m: x[train] for m, x in X.items()})
            X_test = OrderedDict({m: x[test] for m, x in X.items()})
            estimator.fit_(X_train, y[train])
            weighting = OrderedDict({m: combo[i] for i, m in enumerate(X.keys())})
            pred[f] = estimator.predict_(X_test, weighting=weighting).tolist()
            y_true[f] = y[test].tolist()
        score = scoringfunc(sum(y_true, []), sum(pred, []))
        # if not i or not np.mod(i, modulus):
        print('\t[%s] Combo %g / %s %s %.2f' % (time.strftime("%d.%m %H:%M:%S"), i + 1, n_combos, combo, 100 * score))
        return score

    def predict(self, X):
        if self.weight_fixed is not None:
            return self.predict_(X, weighting=self.weight_fixed)
        else:
            return self.predict_(X, weighting=self.weighting)

    def predict_(self, X, weighting=None):
        if not isinstance(X, dict):
            raise ValueError("X has to be a dict")

        n_sample = len(X[list(X.keys())[0]])
        n_estimators = len(self.base_estimators)
        pred = np.full((n_estimators, n_sample), np.nan)
        for i, k in enumerate(X.keys()):
            x = np.array(X[k])
            valid = ~np.isnan(x[:, 0])
            if valid.any():
                if hasattr(self.base_estimators[k], 'predict_proba'):
                    pred[i, valid] = self.base_estimators[k].predict_proba(x[valid])[:, 1] - 0.5
                else:
                    pred[i, valid] = self.base_estimators[k].decision_function(x[valid])
                self.base_estimators[k].pred_modality = pred[i, valid]
            else:
                self.base_estimators[k].pred_modality = np.array([np.nan])

        if self.discrete:
            pred = np.sign(pred)

        if weighting is None:
            pred_multimodal = np.array([self.classes_[int(p>0)] for p in np.nanmean(pred, axis=0)])
        else:
            weights = np.array(list(weighting.values()))
            pred_multimodal = np.array([self.classes_[int(np.average(p[~np.isnan(p)], weights=weights[~np.isnan(p)])>0)] for p in pred.T])
        return pred_multimodal

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        for key, value in params.items():
            datatype, key = key.split('__')
            valid_params = self.base_estimators[datatype].get_params(deep=True)

            nested_params = defaultdict(dict)  # grouped by prefix
            key, delim, sub_key = key.partition('__')
            if key in valid_params:
                if delim:
                    nested_params[key][sub_key] = value
                else:
                    setattr(self.base_estimators[datatype], key, value)
                    valid_params[key] = value

                for key, sub_params in nested_params.items():
                    valid_params[key].set_params(**sub_params)

        return self