"""
The :mod:`sklearn.grid_search` includes utilities to fine-tune the parameters
of an estimator.
"""
from __future__ import print_function

import importlib
import numbers
import time
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from itertools import product

import numpy as np
from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.base import MetaEstimatorMixin
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import check_cv
from sklearn.model_selection._search import _CVScoreTuple, _check_param_grid, ParameterGrid, \
    NotFittedError, check_is_fitted, MaskedArray, defaultdict, rankdata, _aggregate_score_dicts, \
    DeprecationDict, _check_multimetric_scoring
from sklearn.model_selection._validation import _index_param_value, safe_indexing, _score, \
    FitFailedWarning, logger
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import _num_samples, indexable
from collections import Mapping, namedtuple, defaultdict, Sequence
from functools import partial, reduce
from itertools import product
import operator
from collections import OrderedDict
from .grid_search import MultiModalGridSearchCV



class MultiModalSeparateBaseSearchCV(six.with_metaclass(ABCMeta, BaseEstimator,
                                      MetaEstimatorMixin)):
    """Base class for hyper parameter search with cross-validation."""

    @abstractmethod
    def __init__(self, estimator, scoring=None,
                 fit_params=None, n_jobs=1, iid=True,
                 refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs',
                 error_score='raise', return_train_score=True,
                 optimize_on_whole=False, optimize_default=False, optimize_default_loop=False):

        self.scoring = scoring
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.fit_params = fit_params
        self.iid = iid
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.optimize_on_whole = optimize_on_whole
        self.optimize_default = optimize_default
        self.optimize_default_loop = optimize_default_loop

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def score(self, X, y=None):
        """Returns the score on the given data, if the estimator has been refit.

        This uses the score defined by ``scoring`` where provided, and the
        ``best_estimator_.score`` method otherwise.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        score : float
        """
        self._check_is_fitted('score')
        if self.scorer_ is None:
            raise ValueError("No score function explicitly defined, "
                             "and the estimator doesn't provide one %s"
                             % self.best_estimator_)
        score = self.scorer_[self.refit] if self.multimetric_ else self.scorer_
        return score(self.best_estimator_, X, y)

    def _check_is_fitted(self, method_name):
        if not self.refit:
            raise NotFittedError('This %s instance was initialized '
                                 'with refit=False. %s is '
                                 'available only after refitting on the best '
                                 'parameters. You can refit an estimator '
                                 'manually using the ``best_parameters_`` '
                                 'attribute'
                                 % (type(self).__name__, method_name))
        else:
            check_is_fitted(self, 'best_estimator_')

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict(self, X):
        """Call predict on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('predict')
        return self.best_estimator_.predict(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_proba(self, X):
        """Call predict_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_proba``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('predict_proba')
        return self.best_estimator_.predict_proba(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_log_proba(self, X):
        """Call predict_log_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_log_proba``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('predict_log_proba')
        return self.best_estimator_.predict_log_proba(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def decision_function(self, X):
        """Call decision_function on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``decision_function``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('decision_function')
        return self.best_estimator_.decision_function(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def transform(self, X):
        """Call transform on the estimator with the best found parameters.

        Only available if the underlying estimator supports ``transform`` and
        ``refit=True``.

        Parameters
        -----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('transform')
        return self.best_estimator_.transform(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def inverse_transform(self, Xt):
        """Call inverse_transform on the estimator with the best found params.

        Only available if the underlying estimator implements
        ``inverse_transform`` and ``refit=True``.

        Parameters
        -----------
        Xt : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('inverse_transform')
        return self.best_estimator_.inverse_transform(Xt)

    @property
    def classes_(self):
        self._check_is_fitted("classes_")
        return self.best_estimator_.classes_

    def fit(self, X, y=None, groups=None, **fit_params):
        if self.optimize_default_loop:
            return self._fit_default_loop(X, y=y, groups=groups, **fit_params)
        else:
            return self._fit(X, y=y, groups=groups, **fit_params)

    def _fit(self, X, y=None, groups=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        if self.fit_params is not None:
            warnings.warn('"fit_params" as a constructor argument was '
                          'deprecated in version 0.19 and will be removed '
                          'in version 0.21. Pass fit parameters to the '
                          '"fit" method instead.', DeprecationWarning)
            if fit_params:
                warnings.warn('Ignoring fit_params passed as a constructor '
                              'argument in favor of keyword arguments to '
                              'the "fit" method.', RuntimeWarning)
            else:
                fit_params = self.fit_params
        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        scorers, self.multimetric_ = _check_multimetric_scoring(self.estimator, scoring=self.scoring)

        if self.multimetric_:
            if self.refit is not False and (
                    not isinstance(self.refit, six.string_types) or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers):
                raise ValueError("For multi-metric scoring, the parameter "
                                 "refit must be set to a scorer key "
                                 "to refit an estimator with the best "
                                 "parameter setting on the whole data and "
                                 "make the best_* attributes "
                                 "available for that metric. If this is not "
                                 "needed, refit should be set to False "
                                 "explicitly. %r was passed." % self.refit)
            else:
                refit_metric = self.refit
        else:
            refit_metric = 'score'

        clfind = [v[0] for v in self.estimator.steps].index('clf')
        base_estimators = self.estimator.steps[clfind][1].base_estimators
        modalities = list(base_estimators.keys())

        # X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)

        pre_dispatch = self.pre_dispatch

        if self.refit:
            self.best_estimator_ = clone(self.estimator)

        if self.optimize_default:
            gs = MultiModalGridSearchCV(clone(self.estimator), self.param_grid, scoring=self.scoring,
                                        fit_params=self.fit_params, n_jobs=self.n_jobs, iid=self.iid,
                                        refit=False, cv=self.cv, verbose=self.verbose, pre_dispatch=self.pre_dispatch,
                                        error_score=self.error_score, return_train_score=self.return_train_score, cadence=True)
            gs.fit(X, y, groups=groups, **fit_params)
            k0 = list(X.keys())[0]
            default_parameter = {k.replace('clf__%s__' % k0, ''): v for k, v in
                                 gs.cv_results_['params'][gs.best_index_].items() if k0 in k}

        modresults = dict()
        for modality, estimator in base_estimators.items():
            # Regenerate parameter iterable for each fit
            candidate_params = list(self._get_param_iterator(modality))
            n_candidates = len(candidate_params)
            if not (n_candidates == 1 and not candidate_params[0]): # check if grid search necessary for current modality
                base_estimator = clone(estimator)
                if self.verbose > 0:
                    print("[{0}] Fitting {1} folds for each of {2} candidates, totalling"
                          " {3} fits".format(modality, n_splits, n_candidates,
                                             n_candidates * n_splits))

                if self.optimize_on_whole:
                    fit_func = _fit_and_score_multimodal
                    fit_estimator = self.estimator
                    if self.optimize_default:
                        fit_kwargs = dict(default_parameters=default_parameter)
                    else:
                        fit_kwargs = dict(default_parameters=candidate_params[0])
                else:
                    fit_func = _fit_and_score
                    fit_estimator = base_estimator
                    fit_kwargs = dict()

                out = Parallel(
                    n_jobs=self.n_jobs, verbose=self.verbose,
                    pre_dispatch=pre_dispatch
                )(delayed(fit_func)(clone(fit_estimator), X, modality, y, scorers, train,
                                    test, self.verbose, parameters,
                                    return_train_score=self.return_train_score,
                                    return_n_test_samples=True,
                                    return_times=True, return_parameters=False, **fit_kwargs)
                  for parameters, (train, test) in product(candidate_params,
                                                           cv.split(X[modality], y, groups)))

                # if one choose to see train score, "out" will contain train score info
                if self.return_train_score:
                    (train_score_dicts, test_score_dicts, test_sample_counts, fit_time,
                     score_time) = zip(*out)
                else:
                    (test_score_dicts, test_sample_counts, fit_time,
                     score_time) = zip(*out)

                # test_score_dicts and train_score dicts are lists of dictionaries and
                # we make them into dict of lists
                test_scores = _aggregate_score_dicts(test_score_dicts)
                # if self.return_train_score:
                #     train_scores = _aggregate_score_dicts(train_score_dicts)

                # TODO: replace by a dict in 0.21
                results = (DeprecationDict() if self.return_train_score == 'warn'
                           else {})

                def _store(key_name, array, splits=False, rank=False):
                    """A small helper to store the scores/times to the cv_results_"""
                    # When iterated first by splits, then by parameters
                    # We want `array` to have `n_candidates` rows and `n_splits` cols.
                    array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                                      n_splits)
                    if splits:
                        for split_i in range(n_splits):
                            # Uses closure to alter the results
                            results["split%d_%s"
                                    % (split_i, key_name)] = array[:, split_i]

                    array_means = np.nanmean(array, axis=1)
                    results['mean_%s' % key_name] = array_means
                    array_stds = np.sqrt(np.nanmean((array - array_means[:, np.newaxis]) ** 2, axis=1))
                    results['std_%s' % key_name] = array_stds

                    if rank:
                        results["rank_%s" % key_name] = np.asarray(
                            rankdata(-array_means, method='min'), dtype=np.int32)

                # _store('fit_time', fit_time)
                # _store('score_time', score_time)
                # Use one MaskedArray and mask all the places where the param is not
                # applicable for that candidate. Use defaultdict as each candidate may
                # not contain all the params
                param_results = defaultdict(partial(MaskedArray,
                                                    np.empty(n_candidates,),
                                                    mask=True,
                                                    dtype=object))
                for cand_i, params in enumerate(candidate_params):
                    for name, value in params.items():
                        # An all masked empty array gets created for the key
                        # `"param_%s" % name` at the first occurence of `name`.
                        # Setting the value at an index also unmasks that index
                        param_results["param_%s" % name][cand_i] = value

                results.update(param_results)
                # Store a list of param dicts at the key 'params'
                results['params'] = candidate_params

                test_sample_counts = np.array(test_sample_counts[:n_splits], dtype=np.int)
                for scorer_name in scorers.keys():
                    _store('test_%s' % scorer_name, test_scores[scorer_name],
                           splits=True, rank=True)

                # For multi-metric evaluation, store the best_index_, best_params_ and
                # best_score_ iff refit is one of the scorer names
                # In single metric evaluation, refit_metric is "score"
                if self.refit or not self.multimetric_:
                    best_index_ = results["rank_test_%s" % refit_metric].argmin()
                    best_params_ = candidate_params[best_index_]
                    best_score_ = results["mean_test_%s" % refit_metric][best_index_]

                if self.refit:
                    self.best_estimator_.steps[clfind][1].base_estimators[modality].set_params(**best_params_)

                modresults[modality] = results

        if self.refit:
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)


        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers['score']

        self.cv_results_ = modresults
        self.n_splits_ = n_splits

        return self







    def _fit_default_loop(self, X, y=None, groups=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        if self.fit_params is not None:
            warnings.warn('"fit_params" as a constructor argument was '
                          'deprecated in version 0.19 and will be removed '
                          'in version 0.21. Pass fit parameters to the '
                          '"fit" method instead.', DeprecationWarning)
            if fit_params:
                warnings.warn('Ignoring fit_params passed as a constructor '
                              'argument in favor of keyword arguments to '
                              'the "fit" method.', RuntimeWarning)
            else:
                fit_params = self.fit_params
        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        scorers, self.multimetric_ = _check_multimetric_scoring(self.estimator, scoring=self.scoring)

        if self.multimetric_:
            if self.refit is not False and (
                    not isinstance(self.refit, six.string_types) or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers):
                raise ValueError("For multi-metric scoring, the parameter "
                                 "refit must be set to a scorer key "
                                 "to refit an estimator with the best "
                                 "parameter setting on the whole data and "
                                 "make the best_* attributes "
                                 "available for that metric. If this is not "
                                 "needed, refit should be set to False "
                                 "explicitly. %r was passed." % self.refit)
            else:
                refit_metric = self.refit
        else:
            refit_metric = 'score'

        clfind = [v[0] for v in self.estimator.steps].index('clf')
        base_estimators = self.estimator.steps[clfind][1].base_estimators
        modalities = list(base_estimators.keys())

        # X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)

        pre_dispatch = self.pre_dispatch

        if self.refit:
            self.best_estimator_ = clone(self.estimator)


        k0 = list(X.keys())[0]
        default_parameters = list(self._get_param_iterator(k0))

        maxscore = -1
        modresults = [None] * len(default_parameters)
        for i, default_parameter in enumerate(default_parameters):
            print('\t[%s] Default Parameter %g / %g [%s]' % (time.strftime("%d.%m %H:%M:%S"), i + 1, len(default_parameters), default_parameter))
            modresults[i] = dict()
            for modality, estimator in base_estimators.items():

                # Regenerate parameter iterable for each fit
                candidate_params = list(self._get_param_iterator(modality))
                n_candidates = len(candidate_params)
                if self.verbose > 0:
                    print("[{0}] Fitting {1} folds for each of {2} candidates, totalling"
                          " {3} fits".format(modality, n_splits, n_candidates,
                                             n_candidates * n_splits))

                fit_kwargs = dict(default_parameters=default_parameter)

                out = Parallel(
                    n_jobs=self.n_jobs, verbose=self.verbose,
                    pre_dispatch=pre_dispatch
                )(delayed(_fit_and_score_multimodal)(clone(self.estimator), X, modality, y, scorers, train,
                                                     test, self.verbose, parameters,
                                                     return_train_score=self.return_train_score,
                                                     return_n_test_samples=True,
                                                     return_times=True, return_parameters=False, **fit_kwargs)
                  for parameters, (train, test) in product(candidate_params,
                                                           cv.split(X[modality], y, groups)))

                # if one choose to see train score, "out" will contain train score info
                if self.return_train_score:
                    (train_score_dicts, test_score_dicts, test_sample_counts, fit_time,
                     score_time) = zip(*out)
                else:
                    (test_score_dicts, test_sample_counts, fit_time,
                     score_time) = zip(*out)

                # test_score_dicts and train_score dicts are lists of dictionaries and
                # we make them into dict of lists
                test_scores = _aggregate_score_dicts(test_score_dicts)
                # if self.return_train_score:
                #     train_scores = _aggregate_score_dicts(train_score_dicts)

                # TODO: replace by a dict in 0.21
                results = (DeprecationDict() if self.return_train_score == 'warn'
                           else {})

                def _store(key_name, array, splits=False, rank=False):
                    """A small helper to store the scores/times to the cv_results_"""
                    # When iterated first by splits, then by parameters
                    # We want `array` to have `n_candidates` rows and `n_splits` cols.
                    array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                                      n_splits)
                    if splits:
                        for split_i in range(n_splits):
                            # Uses closure to alter the results
                            results["split%d_%s"
                                    % (split_i, key_name)] = array[:, split_i]

                    array_means = np.nanmean(array, axis=1)
                    results['mean_%s' % key_name] = array_means
                    array_stds = np.sqrt(np.nanmean((array - array_means[:, np.newaxis]) ** 2, axis=1))
                    results['std_%s' % key_name] = array_stds

                    if rank:
                        results["rank_%s" % key_name] = np.asarray(
                            rankdata(-array_means, method='min'), dtype=np.int32)

                # _store('fit_time', fit_time)
                # _store('score_time', score_time)
                # Use one MaskedArray and mask all the places where the param is not
                # applicable for that candidate. Use defaultdict as each candidate may
                # not contain all the params
                param_results = defaultdict(partial(MaskedArray,
                                                    np.empty(n_candidates,),
                                                    mask=True,
                                                    dtype=object))
                for cand_i, params in enumerate(candidate_params):
                    for name, value in params.items():
                        # An all masked empty array gets created for the key
                        # `"param_%s" % name` at the first occurence of `name`.
                        # Setting the value at an index also unmasks that index
                        param_results["param_%s" % name][cand_i] = value

                results.update(param_results)
                # Store a list of param dicts at the key 'params'
                results['params'] = candidate_params

                test_sample_counts = np.array(test_sample_counts[:n_splits], dtype=np.int)
                for scorer_name in scorers.keys():
                    _store('test_%s' % scorer_name, test_scores[scorer_name], splits=True, rank=True)

                results['best_index'] = results["rank_test_%s" % refit_metric].argmin()
                results['best_params'] = candidate_params[results['best_index']]
                results['best_score'] = results["mean_test_%s" % refit_metric][results['best_index']]

                modresults[i][modality] = results

            parameters = {m: modresults[i][m]['best_params'] for m in modresults[i].keys()}
            out = Parallel(
                n_jobs=self.n_jobs, verbose=self.verbose,
                pre_dispatch=pre_dispatch
            )(delayed(_fit_and_score_multimodal)(clone(self.estimator), X, 'all', y, scorers, train,
                                                 test, self.verbose, parameters,
                                                 return_train_score=self.return_train_score,
                                                 return_n_test_samples=True,
                                                 return_times=True, return_parameters=False, **fit_kwargs)
              for train, test in cv.split(X[k0], y, groups))
            (train_score_dicts, test_score_dicts, test_sample_counts, fit_time, score_time) = zip(*out)
            score = np.nanmean(_aggregate_score_dicts(test_score_dicts)['score'])
            if score > maxscore:
                best_params = parameters
                best_modresult = modresults[i]

        if self.refit:
            for m in base_estimators.keys():
                self.best_estimator_.steps[clfind][1].base_estimators[m].set_params(**best_params[m])
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)


        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers['score']

        self.cv_results_ = best_modresult
        self.n_splits_ = n_splits

        return self












    @property
    def grid_scores_(self):
        check_is_fitted(self, 'cv_results_')
        if self.multimetric_:
            raise AttributeError("grid_scores_ attribute is not available for"
                                 " multi-metric evaluation.")
        warnings.warn(
            "The grid_scores_ attribute was deprecated in version 0.18"
            " in favor of the more elaborate cv_results_ attribute."
            " The grid_scores_ attribute will not be available from 0.20",
            DeprecationWarning)

        grid_scores = list()

        for i, (params, mean, std) in enumerate(zip(
                self.cv_results_['params'],
                self.cv_results_['mean_test_score'],
                self.cv_results_['std_test_score'])):
            scores = np.array(list(self.cv_results_['split%d_test_score'
                                                    % s][i]
                                   for s in range(self.n_splits_)),
                              dtype=np.float64)
            grid_scores.append(_CVScoreTuple(params, mean, scores))

        return grid_scores


class MultiModalSeparateGridSearchCV(MultiModalSeparateBaseSearchCV):
    """Exhaustive search over specified parameter values for an estimator.

    Important members are fit, predict.

    GridSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.

    Read more in the :ref:`User Guide <grid_search>`.

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        See :ref:`multimetric_grid_search` for an example.

        If None, the estimator's default scorer (if available) is used.

    fit_params : dict, optional
        Parameters to pass to the fit method.

        .. deprecated:: 0.19
           ``fit_params`` as a constructor argument was deprecated in version
           0.19 and will be removed in version 0.21. Pass fit parameters to
           the ``fit`` method instead.

    n_jobs : int, default=1
        Number of jobs to run in parallel.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    refit : boolean, or string, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a string denoting the
        scorer is used to find the best parameters for refitting the estimator
        at the end.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``GridSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_parameters_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : boolean, optional
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.

        Current default is ``'warn'``, which behaves as ``True`` in addition
        to raising a warning when a training score is looked up.
        That default will be changed to ``False`` in 0.21.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.


    Examples
    --------
    >>> from sklearn import svm, datasets
    >>> from sklearn.model_selection import GridSearchCV
    >>> iris = datasets.load_iris()
    >>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    >>> svc = svm.SVC()
    >>> clf = MultiModalSeparateGridSearchCV(svc, parameters)
    >>> clf.fit(iris.data, iris.target)
    ...                             # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    GridSearchCV(cv=None, error_score=...,
           estimator=SVC(C=1.0, cache_size=..., class_weight=..., coef0=...,
                         decision_function_shape='ovr', degree=..., gamma=...,
                         kernel='rbf', max_iter=-1, probability=False,
                         random_state=None, shrinking=True, tol=...,
                         verbose=False),
           fit_params=None, iid=..., n_jobs=1,
           param_grid=..., pre_dispatch=..., refit=..., return_train_score=...,
           scoring=..., verbose=...)
    >>> sorted(clf.cv_results_.keys())
    ...                             # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ['mean_fit_time', 'mean_score_time', 'mean_test_score',...
     'mean_train_score', 'param_C', 'param_kernel', 'params',...
     'rank_test_score', 'split0_test_score',...
     'split0_train_score', 'split1_test_score', 'split1_train_score',...
     'split2_test_score', 'split2_train_score',...
     'std_fit_time', 'std_score_time', 'std_test_score', 'std_train_score'...]

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +------------+-----------+------------+-----------------+---+---------+
        |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_t...|
        +============+===========+============+=================+===+=========+
        |  'poly'    |     --    |      2     |        0.8      |...|    2    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'poly'    |     --    |      3     |        0.7      |...|    4    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.1   |     --     |        0.8      |...|    3    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.2   |     --     |        0.9      |...|    1    |
        +------------+-----------+------------+-----------------+---+---------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                         mask = [False False False False]...)
            'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                        mask = [ True  True False False]...),
            'param_degree': masked_array(data = [2.0 3.0 -- --],
                                         mask = [False False  True  True]...),
            'split0_test_score'  : [0.8, 0.7, 0.8, 0.9],
            'split1_test_score'  : [0.82, 0.5, 0.7, 0.78],
            'mean_test_score'    : [0.81, 0.60, 0.75, 0.82],
            'std_test_score'     : [0.02, 0.01, 0.03, 0.03],
            'rank_test_score'    : [2, 4, 3, 1],
            'split0_train_score' : [0.8, 0.9, 0.7],
            'split1_train_score' : [0.82, 0.5, 0.7],
            'mean_train_score'   : [0.81, 0.7, 0.7],
            'std_train_score'    : [0.03, 0.03, 0.04],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
            'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    Notes
    ------
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    ---------
    :class:`ParameterGrid`:
        generates all the combinations of a hyperparameter grid.

    :func:`sklearn.model_selection.train_test_split`:
        utility function to split the data into a development set usable
        for fitting a GridSearchCV instance and an evaluation set for
        its final evaluation.

    :func:`sklearn.metrics.make_scorer`:
        Make a scorer from a performance metric or loss function.

    """

    def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise',
                 return_train_score="warn",
                 optimize_on_whole=False, optimize_default=False, optimize_default_loop=False):
        super(MultiModalSeparateGridSearchCV, self).__init__(
            estimator=estimator, scoring=scoring, fit_params=fit_params,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score,
            optimize_on_whole=optimize_on_whole, optimize_default=optimize_default, optimize_default_loop=optimize_default_loop)
        if optimize_default and optimize_default_loop:
            raise Exception('Parameters optimize_default and optimize_default_loop cannot both be true')
        self.param_grid = param_grid
        _check_param_grid(param_grid)

    def _get_param_iterator(self, modality):
        """Return ParameterGrid instance for the given param_grid"""
        return ParameterGridSeparate(self.param_grid, modality)


class ParameterGridSeparate(object):
    """Grid of parameters with a discrete number of values for each.

    Can be used to iterate over parameter value combinations with the
    Python built-in function iter.

    Read more in the :ref:`User Guide <search>`.

    Parameters
    ----------
    param_grid : dict of string to sequence, or sequence of such
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.

        An empty dict signifies default parameters.

        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See the examples below.

    Examples
    --------
    >>> from sklearn.model_selection import ParameterGrid
    >>> param_grid = {'a': [1, 2], 'b': [True, False]}
    >>> list(ParameterGrid(param_grid)) == (
    ...    [{'a': 1, 'b': True}, {'a': 1, 'b': False},
    ...     {'a': 2, 'b': True}, {'a': 2, 'b': False}])
    True

    >>> grid = [{'kernel': ['linear']}, {'kernel': ['rbf'], 'gamma': [1, 10]}]
    >>> list(ParameterGrid(grid)) == [{'kernel': 'linear'},
    ...                               {'kernel': 'rbf', 'gamma': 1},
    ...                               {'kernel': 'rbf', 'gamma': 10}]
    True
    >>> ParameterGrid(grid)[1] == {'kernel': 'rbf', 'gamma': 1}
    True

    See also
    --------
    :class:`GridSearchCV`:
        Uses :class:`ParameterGrid` to perform a full parallelized parameter
        search.
    """

    def __init__(self, param_grid, modality):
        if isinstance(param_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_grid = [param_grid]
        self.param_grid = param_grid
        # [{k: v for k, v in p.items() if k.split('__')[1] == modality} for p in self.param_grid]
        self.modality = modality

    def __iter__(self):
        """Iterate over the points in the grid.

        Returns
        -------
        params : iterator over dict of string to any
            Yields dictionaries mapping each estimator parameter to one of its
            allowed values.
        """
        for p in self.param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            modstr = '%s__' % self.modality
            items = sorted([(k.replace('clf__'+modstr, ''), v) for k, v in p.items() if modstr in k])
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        """Number of points on the grid."""
        # Product function that can handle iterables (np.product can't).
        product = partial(reduce, operator.mul)
        modstr = '%s__' % self.modality
        return sum(product(len(v) for k, v in p.items() if modstr in k) if p else 1
                   for p in self.param_grid)

    def __getitem__(self, ind):
        """Get the parameters that would be ``ind``th in iteration

        Parameters
        ----------
        ind : int
            The iteration index

        Returns
        -------
        params : dict of string to any
            Equal to list(self)[ind]
        """
        # This is used to make discrete sampling without replacement memory
        # efficient.
        1 / 0
        for sub_grid in self.param_grid:
            # XXX: could memoize information used here
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue

            # Reverse so most frequent cycling parameter comes first
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes)

            if ind >= total:
                # Try the next grid
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out

        raise IndexError('ParameterGrid index out of range')



def _fit_and_score(estimator, X, modality, y, scorer, train, test, verbose,
                   parameters, return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=False):
    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.

        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.

        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.

    train : array-like, shape (n_train_samples,)
        Indices of training samples.

    test : array-like, shape (n_test_samples,)
        Indices of test samples.

    verbose : integer
        The verbosity level.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    parameters : dict or None
        Parameters to be set on the estimator.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    return_train_score : boolean, optional, default: False
        Compute and return score on training set.

    return_parameters : boolean, optional, default: False
        Return parameters that has been used for the estimator.

    return_n_test_samples : boolean, optional, default: False
        Whether to return the ``n_test_samples``

    return_times : boolean, optional, default: False
        Whether to return the fit/score times.

    Returns
    -------
    train_scores : dict of scorer name -> float, optional
        Score on training set (for all the scorers),
        returned only if `return_train_score` is `True`.

    test_scores : dict of scorer name -> float, optional
        Score on testing set (for all the scorers).

    n_test_samples : int
        Number of test samples.

    fit_time : float
        Time spent for fitting in seconds.

    score_time : float
        Time spent for scoring in seconds.

    parameters : dict or None, optional
        The parameters that have been evaluated.
    """

    X = X[modality]

    # Adjust length of sample weights
    # fit_params = fit_params if fit_params is not None else {}
    # fit_params = dict([(k, _index_param_value(X, v, train))
    #                   for k, v in fit_params.items()])

    train_scores = {}
    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    valid_train = [i for i, x in enumerate(X_train) if ~np.any(np.isnan(x))]
    X_train = [x for i, x in enumerate(X_train) if i in valid_train]
    y_train = [y_ for i, y_ in enumerate(y_train) if i in valid_train]
    valid_test = [i for i, x in enumerate(X_test) if ~np.any(np.isnan(x))]
    X_test = [x for i, x in enumerate(X_test) if i in valid_test]
    y_test = [y_ for i, y_ in enumerate(y_test) if i in valid_test]

    is_multimetric = not callable(scorer)

    if y_train is None:
        # estimator.fit(X_train, **fit_params)
        estimator.fit(X_train)
    else:
        # estimator.fit(X_train, y_train, **fit_params)
        estimator.fit(X_train, y_train)

    fit_time = time.time() - start_time
    # _score will return dict if is_multimetric is True
    if y_test:
        test_scores = _score(estimator, X_test, y_test, scorer, is_multimetric)
    else:
        test_scores = dict(score=np.nan)

    score_time = time.time() - start_time - fit_time
    if return_train_score:
        train_scores = _score(estimator, X_train, y_train, scorer,
                              is_multimetric)

    ret = [train_scores, test_scores] if return_train_score else [test_scores]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    return ret


def _fit_and_score_multimodal(estimator, X, modality, y, scorer, train, test, verbose,
                              parameters, return_train_score=False,
                              return_parameters=False, return_n_test_samples=False,
                              return_times=False, default_parameters=None):
    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.

        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.

        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.

    train : array-like, shape (n_train_samples,)
        Indices of training samples.

    test : array-like, shape (n_test_samples,)
        Indices of test samples.

    verbose : integer
        The verbosity level.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    parameters : dict or None
        Parameters to be set on the estimator.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    return_train_score : boolean, optional, default: False
        Compute and return score on training set.

    return_parameters : boolean, optional, default: False
        Return parameters that has been used for the estimator.

    return_n_test_samples : boolean, optional, default: False
        Whether to return the ``n_test_samples``

    return_times : boolean, optional, default: False
        Whether to return the fit/score times.

    Returns
    -------
    train_scores : dict of scorer name -> float, optional
        Score on training set (for all the scorers),
        returned only if `return_train_score` is `True`.

    test_scores : dict of scorer name -> float, optional
        Score on testing set (for all the scorers).

    n_test_samples : int
        Number of test samples.

    fit_time : float
        Time spent for fitting in seconds.

    score_time : float
        Time spent for scoring in seconds.

    parameters : dict or None, optional
        The parameters that have been evaluated.
    """
    if verbose > 1 and modality != 'all':
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                          for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    y = np.array(y)

    # Adjust length of sample weights
    # fit_params = fit_params if fit_params is not None else {}
    # fit_params = dict([(k, _index_param_value(X, v, train))
    #                   for k, v in fit_params.items()])

    train_scores = {}

    clfind = [v[0] for v in estimator.steps].index('clf')
    if modality == 'all':
        for k in estimator.steps[clfind][1].base_estimators.keys():
            estimator.steps[clfind][1].base_estimators[k].set_params(**parameters[k])
    elif parameters is not None:
        for k in estimator.steps[clfind][1].base_estimators.keys():
            if k == modality:
                estimator.steps[clfind][1].base_estimators[k].set_params(**parameters)
            else:
                estimator.steps[clfind][1].base_estimators[k].set_params(**default_parameters)


    start_time = time.time()

    y_train = y[train]
    y_test = y[test]

    X_train = OrderedDict({k: np.array(x)[train] for k, x in X.items()})
    X_test = OrderedDict({k: np.array(x)[test] for k, x in X.items()})
    # X_train, X_test = dict(), dict()
    # for k, X_ in X.items():
    #     x_train, x_test = X_[train], X_[test]
    #     valid_train = [i for i, x in enumerate(x_train) if ~np.any(np.isnan(x))]
    #     X_train[k] = [x for i, x in enumerate(x_train) if i in valid_train]
    #     valid_test = [i for i, x in enumerate(x_test) if ~np.any(np.isnan(x))]
    #     X_test[k] = [x for i, x in enumerate(x_test) if i in valid_test]

    is_multimetric = not callable(scorer)

    if y_train is None:
        # estimator.fit(X_train, **fit_params)
        estimator.fit(X_train)
    else:
        # estimator.fit(X_train, y_train, **fit_params)
        estimator.fit(X_train, y_train)

    fit_time = time.time() - start_time
    # _score will return dict if is_multimetric is True
    test_scores = _score(estimator, X_test, y_test, scorer, is_multimetric)
    score_time = time.time() - start_time - fit_time
    if return_train_score:
        train_scores = _score(estimator, X_train, y_train, scorer,
                              is_multimetric)

    if verbose > 2:
        if is_multimetric:
            for scorer_name, score in test_scores.items():
                msg += ", %s=%s" % (scorer_name, score)
        else:
            msg += ", score=%s" % test_scores
    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = "%s, total=%s" % (msg, logger.short_format_time(total_time))
        print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

    ret = [train_scores, test_scores] if return_train_score else [test_scores]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    return ret






def _safe_split_multimodal(estimator, X, y, indices, train_indices=None):

    X_subset = OrderedDict({k: [v[i] for i in indices] for k, v in X.items()})

    if y is not None:
        y_subset = safe_indexing(y, indices)
    else:
        y_subset = None

    return X_subset, y_subset
