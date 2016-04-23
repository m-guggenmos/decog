import time
import timeit
import importlib
import warnings
from inspect import signature
import multiprocessing_on_dill as multiprocessing
import numpy as np
from scipy.stats import binom_test

from sklearn.cross_validation import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression, \
    VarianceThreshold, SelectFromModel
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, r2_score, explained_variance_score, \
    confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from treeinterpreter import treeinterpreter as ti

from .feature_selection import MultiRoiVarianceThreshold, MuliRoiSelectPercentile, \
    MultiRoiSelectFromModel, SelectRoisFromModel, MultiRoiVariancePercentile, VariancePercentile
from .cv import DummyCV
import pickle


class Analysis:

    def __init__(self, scheme, name=''):
        """ Run a single analysis according to the provided processing scheme

        Parameters
        ----------
        scheme : decereb.descriptor.SchemeDescriptor
        name : str
            Optional identyfying for the analysis
        """
        self.scheme = scheme
        self.name = name

        self.n_channels = len(self.scheme.data)
        self.is_multichannel = self.n_channels > 1
        self.searchlight = self.scheme.channels[0].clfs._searchlight

        self.n_folds = None
        self.cv = None
        self.n_seeds = None
        self.n_samples = None
        self.seed_list = None
        self.is_regression = None
        self.y_true = None
        self.label_names = None
        self.scoring = None

        self.masker = None
        self.selection = None
        self.clf = None
        self.pipeline = None
        self.steps = None
        self.param_grid = None

        self.verbose = None

    def run(self, n_jobs_folds=1, verbose=2):

        """ Start the analysis.

        Parameters
        ----------
        n_jobs_folds : int
            Number of processors for cross-validation
        verbose : int
            verbosity level

        Returns
        -------
        dict

        """

        self._checks()

        self.verbose = verbose
        self.n_jobs_folds = n_jobs_folds
        self.is_regression = self.scheme.channels[0].clfs.regression
        self.y_true = np.array(self.scheme.data[0].labels)
        self.label_names = self.scheme.data[0].label_names
        self.has_time = self.scheme.data[0].has_time

        self.seed_lists = [channel.clfs.seed_list for channel in self.scheme.channels]
        self.n_seeds = [len(sl) for sl in self.seed_lists]
        self.n_samples = len(self.scheme.data[0].labels)

        if not self.searchlight:
            if not hasattr(self.scheme, 'cv') or self.scheme.cv is None:
                self.cv = LeaveOneOut(self.n_samples)
                # from sklearn.cross_validation import KFold
                # self.cv_outer = KFold(self.n_samples)
            else:
                self.cv = self.scheme.cv
        else:  # Nilearn searchlight algorihm uses an internal CV, so here we use a dummy CV
            self.cv = DummyCV(self.n_samples)

        self.n_folds = len(self.cv)

        self._construct_pipe()

        kwargs = dict()
        if self.has_time:
            func_fold = self._fold_time
            if isinstance(self.scheme.data[0].data, str):
                kwargs['X'] = pickle.load(open(self.scheme.data[0].data, 'rb'))
            else:
                kwargs['X'] = self.scheme.data[0].data
        else:
            func_fold = self._fold

        if n_jobs_folds == 1:
            results = []
            for f, (train_index, test_index) in enumerate(self.cv):
                results.append(func_fold((f, train_index, test_index, kwargs)))
        else:
            pool = multiprocessing.Pool(None if n_jobs_folds == -1 else n_jobs_folds)
            results = pool.map(func_fold,
                               [(f, train_index, test_index, kwargs)
                                for f, (train_index, test_index) in enumerate(self.cv)])
            pool.close()

        if self.has_time:
            result = np.mean(results, axis=0)
        elif not self.searchlight:
            result = self._post_processing(results)
            self._print_result(result)
        else:
            result = {'seed%g' % s: results[0][0][0][s]['searchlight']
                      for s in range(self.n_seeds[0])}
            print('Searchlight finished!')

        return result

    def _fold_time(self, params):

        f, train_indices, test_indices, kwargs = params

        X = kwargs['X']

        n_time = X.shape[2]
        n_channels = X.shape[1]

        if self.verbose > 1:
            tic = timeit.default_timer()
            print('[%s] Permutation %g / %g' %
                  (time.strftime("%d.%m %H:%M:%S"), f + 1, self.n_folds))

        Xpseudo_train = np.full((self.cv.n_classes*(self.cv.n_pseudo-1), n_channels,
                                 n_time), np.nan)
        Xpseudo_test = np.full((self.cv.n_classes, n_channels, n_time), np.nan)
        result = np.full((self.cv.n_classes, self.cv.n_classes, n_time), np.nan)

        for i, order_ in enumerate(train_indices):
            Xpseudo_train[i, :, :] = np.mean(X[order_, :, :], axis=0)
        for i, order_ in enumerate(test_indices):
            Xpseudo_test[i, :, :] = np.mean(X[order_, :, :], axis=0)

        # n_comb = (self.cv.n_classes**2 - self.cv.n_classes) / 2
        # k = 1
        for c1 in range(self.cv.n_classes):
            for c2 in range(c1 + 1, self.cv.n_classes):
                # tic2 = timeit.default_timer()
                for time_point in range(n_time):
                    data_train = Xpseudo_train[self.cv.ind_pseudo_train[c1, c2], :, time_point]
                    data_test = Xpseudo_test[self.cv.ind_pseudo_test[c1, c2], :, time_point]
                    self.clf[0].fit(data_train, self.cv.labels_pseudo_train[c1, c2])
                    if self.clf[0]._estimator_type == 'distance':
                        dissimilarity = self.clf[0].predict(data_test, y=self.cv.labels_pseudo_test[c1, c2])
                    else:
                        predictions = self.clf[0].predict(data_test)
                        dissimilarity = np.mean(predictions == self.cv.labels_pseudo_test[c1, c2])
                    result[c1, c2, time_point] = dissimilarity
                # print('Pair %g / %g: %.4s secs' % (k, n_comb, timeit.default_timer() - tic2))
                # k += 1

        if self.verbose > 1:
            print('Permutation time: %.2f secs' % (timeit.default_timer() - tic))

        return result

    def _fold(self, params):
        """

        Parameters
        ----------
        params : tuple
            tuple of the form (fold_index, training indices, testing indices)

        Returns
        -------
        tuple: (channel result, meta result)

        """
        f, train_indices, test_indices, kwargs = params
        n_test = len(test_indices)

        ### Prepare data structures ###
        train_cache = [None] * self.n_channels
        test_cache = [None] * self.n_channels
        result_channels = [[None] * self.n_seeds[c] for c in range(self.n_channels)]
        for c in range(self.n_channels):
            for s in range(self.n_seeds[c]):
                result_channels[c][s] = dict()
                if self.scheme.is_multiroi[c] and not self.searchlight:
                    n_rois = sum([len(m) for m in self.scheme.masker_args[c]['rois']]) \
                        if isinstance(self.scheme.masker_args[c]['rois'], tuple) \
                        else len(self.scheme.masker_args[c]['rois'])
                    result_channels[c][s].update(
                        votes=np.full((n_rois, n_test), np.nan),
                    )
        result_meta = dict()
        ###

        for c in range(self.n_channels):

            X = self.scheme.data[c].data

            for s, seed_ in enumerate(self.seed_lists[c]):

                if self.searchlight:
                    print('[%s] Starting searchlight analysis' % time.strftime("%d.%m %H:%M:%S"))
                elif self.verbose > 1:
                    print('[%s] Fold %g / %g Channel %g / %g Seed %g / %g' %
                          (time.strftime("%d.%m %H:%M:%S"),
                           f + 1, self.n_folds,
                           c + 1, self.n_channels,
                           s + 1, self.n_seeds[c]))
                    if self.name:
                        print('\tof << %s >>' % self.name)

                data_train = [X[i] for i in train_indices]
                labels_train = [self.y_true[i] for i in train_indices]
                data_test = [X[i] for i in test_indices]

                ### (1 / 2) If no GridSearchCV or feature selection (with varying seeds) is used, we
                ### can prefit the preprocessing steps
                if self.param_grid[c] is None and train_cache[c] is None and \
                    not ('fs_model' in dict(self.steps[c]) and seed_ is not None):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        train_cache[c] = Pipeline(self.steps[c][:-1]).fit_transform(data_train,
                                                                                    labels_train)
                ###

                ### Set seeds ###
                if hasattr(self.clf[c], 'random_state'):
                    self.clf[c].random_state = seed_
                if 'fs_model' in dict(self.steps[c]):
                    dict(self.steps[c])['fs_model'].random_state = seed_
                    if hasattr(dict(self.steps[c])['fs_model'].estimator, 'random_state'):
                        dict(self.steps[c])['fs_model'].estimator.random_state = seed_
                    if hasattr(dict(self.steps[c])['fs_model'], 'estimator_') and \
                            hasattr(dict(self.steps[c])['fs_model'].estimator_, 'random_state'):
                        dict(self.steps[c])['fs_model'].estimator_.random_state = seed_
                ###

                ### Fit training data ###
                if self.param_grid[c] is None:
                    if train_cache[c] is not None:
                        self.clf[c].fit(train_cache[c], labels_train)
                    else: # (2 / 2) else we have to fit the entire pipeline for each seed
                        self.pipeline[c].fit(data_train, labels_train)
                else:  # (2 / 2) else we have to fit the entire pipeline for each seed
                    self.pipeline[c].cv = LeaveOneOut(len(train_indices))

                    # otherwise GridSearchCV won't be silent
                    ind_clf = [el[0] for el in self.pipeline[c].estimator.steps].index('clf')
                    self.pipeline[c].estimator.steps[ind_clf][1].verbose = self.pipeline[c].verbose

                    self.pipeline[c].fit(data_train, labels_train)

                    for step in range(len(self.steps[c])):
                        self.steps[c][step] = self.pipeline[c].best_estimator_.steps[step]
                    if len(self.selection[c]):
                        step_classes = [step[1].__class__
                                        for step in self.pipeline[c].best_estimator_.steps]
                        for i, sel in enumerate(self.selection[c]):
                            self.selection[c][i] = self.pipeline[c].best_estimator_.steps[
                                step_classes.index(sel.__class__)][1]
                    self.clf[c] = self.pipeline[c].best_estimator_._final_estimator

                    Pipeline(self.steps[c][:-1]).transform(data_train)
                ###

                ### Transform test data ###
                if test_cache[c] is None or \
                    ('fs_model' in dict(self.steps[c]) and seed_ is not None):
                    test_cache[c] = Pipeline(self.steps[c][:-1]).transform(data_test)
                ###

                ### Prediction ###
                predictions = self.clf[c].predict(test_cache[c])
                if self.is_multichannel and not self.is_regression:
                    # for multi-channel classification problems we try to obtain graded predictions
                    if hasattr(self.clf[c], 'predict_graded'):
                        predictions_graded = self.clf[c].predict_graded(test_cache[c])
                    else:
                        predictions_graded = predictions
                ###

                ### Collect GridSearchCV results ###
                if self.param_grid[c] is not None:
                    result_channels[c][s]['grid'] = \
                        [str(score.parameters) for score in self.pipeline[c].grid_scores_]
                    result_channels[c][s]['grid_scores'] = \
                        [score.mean_validation_score for score in self.pipeline[c].grid_scores_]
                    if self.verbose > 2:
                        for grid, score in zip(result_channels[c][s]['grid'],
                                               result_channels[c][s]['grid_scores']):
                            print('%s: %.5f' % (grid, score))
                ###

                ### Collect RandomForest feature imporances ###
                result_channels = self._forest_importances(X, test_cache, result_channels, c, s)
                ###

                ### Store predictions ###
                if not self.searchlight:
                    result_channels[c][s]['y_pred'] = predictions.astype(float)
                    if self.is_multichannel and not self.is_regression:
                        result_channels[c][s]['y_pred_graded'] = predictions_graded.astype(float)
                    if hasattr(self.clf[c], 'votes_pooled'):
                        result_channels[c][s]['votes_pooled'] = self.clf[c].votes_pooled
                    if hasattr(self.clf[c], 'votes'):
                        for i, k in enumerate(self.clf[c].estimators_.keys()):
                            result_channels[c][s]['votes'][k] = np.array(self.clf[c].votes)[i, :]
                else:
                    result_channels[c][s]['searchlight'] = predictions
                ###

        if self.is_multichannel:
            result_meta['y_pred'] = self.meta_predict(
                [np.mean([s[('y_pred_graded', 'y_pred')[self.is_regression]][test_indices] for s in ch], axis=0)
                 for i, ch in enumerate(result_channels)])

        return result_channels, result_meta

    def _post_processing(self, results):

        """ Post processing routine, computing various measures of the classification performance.

        Parameters
        ----------
        results : List<(channel result, meta result)>

        Returns
        -------
        dict
            dictionary containing the post-processed results

        """
        result_channels = [[None] * self.n_seeds[c] for c in range(self.n_channels)]
        for c in range(self.n_channels):
            for s in range(self.n_seeds[c]):
                result_channels[c][s] = dict(
                    y_pred=np.full(self.n_samples, np.nan),
                    grid_scores=[None] * self.n_folds
                )
                if self.is_multichannel and not self.is_regression:
                    result_channels[c][s].update(
                        y_pred_graded=np.full(self.n_samples, np.nan)
                    )
                if self.scheme.is_multiroi[c]:
                    result_channels[c][s].update(
                        votes=np.full((len(self.scheme.masker_args[0]['rois']), self.n_samples),
                                      np.nan),
                        votes_pooled=np.full(self.n_samples, np.nan),
                    )
        if self.is_multichannel:
            result_meta = dict(
                y_pred=np.full(self.n_samples, np.nan)
            )

        for f, (train_index, test_index) in enumerate(self.cv):
            result_channels_fold, result_meta_fold = results[f]
            if self.is_multichannel:
                result_meta['y_pred'][test_index] = result_meta_fold['y_pred']
            for c in range(self.n_channels):
                is_forest = hasattr(self.clf[c], 'base_estimator') and \
                            isinstance(self.clf[c].base_estimator, (RandomForestRegressor,
                                                                    RandomForestClassifier))
                for s in range(self.n_seeds[c]):
                    result_channels[c][s]['y_pred'][test_index] = \
                        result_channels_fold[c][s]['y_pred']
                    if self.scheme.is_multiroi[c]:
                        result_channels[c][s]['votes'][:, test_index] = \
                            result_channels_fold[c][s]['votes']
                        result_channels[c][s]['votes_pooled'][test_index] = \
                            result_channels_fold[c][s]['votes_pooled']
                    if self.param_grid[c] is not None:
                        result_channels[c][s]['grid_scores'][f] = \
                            result_channels_fold[c][s]['grid_scores']
                    if is_forest and isinstance(self.scheme.data[c].data, np.ndarray):
                        if 'feature_importance' not in result_channels[c][s]:
                            result_channels[c][s]['feature_importance'] = \
                                np.full((self.n_folds,
                                         len(result_channels_fold[c][s]['feature_importance'])),
                                        np.nan)
                        result_channels[c][s]['feature_importance'][f, :] = \
                            result_channels_fold[c][s]['feature_importance']
                    elif self.scheme.is_multiroi[c] and is_forest:
                        if 'feature_importance' not in result_channels[c][s]:
                            result_channels[c][s]['feature_importance'] = dict()
                        for k in result_channels[0][c]['feature_importance'].keys():
                            if k not in result_channels[c][s]['feature_importance']:
                                result_channels[c][s]['feature_importance'][k] = \
                                    np.full((self.n_folds, len(result_channels_fold[c][s]
                                                               ['feature_importance'][k])), np.nan)
                            result_channels[c][s]['feature_importance'][k][f, :] = \
                                result_channels_fold[c][s]['feature_importance'][k]

        result = dict()
        for c in range(self.n_channels):

            is_forest = hasattr(self.clf[c], 'base_estimator') and \
                            isinstance(self.clf[c].base_estimator, (RandomForestRegressor,
                                                                    RandomForestClassifier))

            pfx = '%g_' % c if self.is_multichannel else ''
            result = self._assess_performance(pfx, result_channels[c], container=result)

            if hasattr(self.clf[c], 'votes'):
                result[pfx + 'votes'] = \
                    np.nanmean([result_channels[c][s]['votes']
                                for s in range(self.n_seeds[c])], axis=0).tolist()
            if hasattr(self.clf[c], 'votes_pooled'):
                result[pfx + 'votes_pooled'] = \
                    np.mean([result_channels[c][s]['votes_pooled']
                             for s in range(self.n_seeds[c])], axis=0).tolist()

            if is_forest and isinstance(self.scheme.data[c].data, np.ndarray):
                with warnings.catch_warnings():  # catch stupid behavior of nanmean
                    warnings.simplefilter("ignore", RuntimeWarning)
                    result[pfx + 'forest_contrib'] = \
                        np.nanmean([result_channels[c][s]['feature_importance']
                                    for s in range(self.n_seeds[c])], axis=(0, 1)).tolist()
            elif self.scheme.is_multiroi[c] and is_forest:
                result[pfx + 'forest_contrib'] = {}
                with warnings.catch_warnings():  # catch stupid behavior of nanmean
                    warnings.simplefilter("ignore", RuntimeWarning)
                    for k in result_channels[0][c]['feature_importance'].keys():
                        result[pfx + 'forest_contrib'][k] = \
                            np.mean([result_channels[c][s]['feature_importance'][k]
                                     for s in range(self.n_seeds[c])], axis=(0, 1)).tolist()

            if self.param_grid[c] is not None:
                grid_scores_mean = \
                    np.mean([result_channels[c][s]['grid_scores']
                             for s in range(self.n_seeds[c])], axis=(0, 1)).tolist()
                grid_scores_ste = \
                    (np.std(np.mean([result_channels[c][s]['grid_scores']
                                     for s in range(self.n_seeds[c])], axis=1), axis=0) /
                     self.n_folds).tolist()
                result[pfx + 'grid_scores'] = \
                    [(param, grid_scores_mean[i], grid_scores_ste[i])
                     for i, param in
                     enumerate([scr.parameters for scr in self.pipeline[c].grid_scores_])]

        return result

    def _print_result(self, result):

        """

        Parameters
        ----------
        result : dict
            Dictionary containing the results
        """

        if self.verbose > 0:
            print('***************************************************')
            for c in range(self.n_channels):
                pfx = '%g_' % c if self.is_multichannel else ''

                print('***********       Channel (%g / %g)       ***********' %
                      (c + 1, self.n_channels))
                print('Scheme: %s' % self.scheme.name)
                print("%s: %.5f +- %.5f %s" %
                      (str(self.scoring).split(' ')[1], result[pfx + 'scoring'],
                       result[pfx + 'scoring_ste'],
                      ['%.5f' % acc for acc in result[pfx + 'scoring_seed']]))
                if self.is_regression:
                    print('Explained variance: %.4f%%' %
                          (100 * result[pfx + 'explained_variance']))
                else:
                    print('p = %s' % result[pfx + 'binom_statistic'])
                    print('p(cor.) = %s' % result[pfx + 'binom_statistic_corrected'])
                    print(result[pfx + 'classification_report'])

                # if verbose > 1:
                if hasattr(self.clf[c], 'votes'):
                    print('votes:', *result[pfx + 'votes'] , sep='\n')
                if hasattr(self.clf[c], 'votes_pooled'):
                    print('votes pooled:', result[pfx + 'votes_pooled'] )
                if self.param_grid[c] is not None:
                    for param, av, ste in result[pfx + 'grid_scores']:
                        print('%s: %.5f +- %.5f' % (param, av, ste))
            if self.is_multichannel:
                print('***********       Combined Results       ***********')
                print("%s: %.5f +- %.5f %s" %
                      (str(self.scoring).split(' ')[1],
                       result['scoring'], result['scoring_ste'],
                       ['%.5f' % acc for acc in result['scoring_seed']]))
                if self.is_regression:
                    print('Explained variance: %.4f%%' % 100 * result['explained_variance'])
                else:
                    print(result['classification_report'])
            if self.name:
                print('\tof << %s >>' % self.name)
            print('***************************************************')

        if self.is_multichannel:
            result = self._assess_performance('', [result['meta']], container=result)
            print('meta:', result['predictions'])

    def _assess_performance(self, pfx, result, container=None):
        """ Computes performance measures for the model predictions

        Parameters
        ----------
        pfx : str
            specifies the result type ('' for a case with only one channel or for the
            channel-averaged results, '%g_' for channel-specific results)
        result : List<dict>
            List of result dictionaries for each seed
        container : dict, None
            Optionally provide a dictionary in which the performance assessement should be stored.

        Returns
        -------
        dict
            dictionary filled with performance measures

        """
        n_seeds = len(result)
        if container is None:
            container = dict()
        container[pfx + 'predictions'] = \
            np.mean([result[s]['y_pred'] for s in range(n_seeds)], axis=0).tolist()

        container[pfx + 'scoring_seed'] = \
            [self.scoring(self.y_true, result[s]['y_pred'] if self.is_regression
             else np.round(result[s]['y_pred'])) for s in range(n_seeds)]
        container[pfx + 'scoring_ste'] = \
            np.std(container[pfx + 'scoring_seed'] / np.sqrt(n_seeds)).tolist()
        container[pfx + 'scoring'] = \
            self.scoring(self.y_true, np.round(container[pfx + 'predictions'])).tolist()

        if self.is_regression:
            container[pfx + 'correct'] = None
            container[pfx + 'explained_variance'] = \
                explained_variance_score(self.y_true, container[pfx + 'predictions']).tolist()
        else:
            container[pfx + 'correct'] = (container[pfx + 'predictions'] == self.y_true).tolist()
            p_binom = max([np.mean(self.scheme.data[0].labels == l)
                     for l in np.unique(self.scheme.data[0].labels)])
            container[pfx + 'binom_statistic'] = \
                [binom_test(sum(result[s]['y_pred'] == self.y_true), self.n_samples)
                 for s in range(n_seeds)]
            container[pfx + 'binom_statistic_corrected'] = \
                [binom_test(sum(result[s]['y_pred'] == self.y_true), self.n_samples, p=p_binom)
                 for s in range(n_seeds)]
            container[pfx + 'confusion_matrix'] = \
                confusion_matrix(self.y_true, np.round(container[pfx + 'predictions'])).tolist()
            container[pfx + 'classification_report'] = \
                classification_report(self.y_true, np.round(container[pfx + 'predictions']),
                                      target_names=self.label_names,
                                      labels=np.unique(self.scheme.data[0].labels))

        return container

    def _forest_importances(self, X, test_cache, result_channels, c, s):
        if isinstance(self.clf[c], (RandomForestRegressor, RandomForestClassifier)) and \
                isinstance(X, np.ndarray):
            features = np.ones(X.shape[1], dtype=bool)
            if 'feature_importance' not in result_channels[c][s]:
                result_channels[c][s]['feature_importance'] = np.full(X.shape[1], np.nan)
            for sel in self.selection[c]:
                features[features] = sel.get_support()
            contrib = ti.predict(self.clf[c], np.array(test_cache[c]))[2]
            result_channels[c][s]['feature_importance'][features] = \
                np.mean(contrib if self.is_regression else contrib[:, :, 0], axis=0)
        elif self.scheme.is_multiroi[c] and \
                isinstance(self.clf[c].base_estimator, (RandomForestRegressor,
                                                        RandomForestClassifier)):
            if 'feature_importance' not in result_channels[c][s]:
                result_channels[c][s]['feature_importance'] = dict()
            if sum(self.selection, []):
                features = dict()
                for sel in self.selection[c]:
                    for k, v in sel.get_support().items():
                        if k in test_cache[c].keys():
                            if k not in features:
                                features[k] = v
                            else:
                                features[k][features[k]] = v
            else:
                features = {k: np.ones(len(v[0]), dtype=bool)
                            for k, v in test_cache[c].items()}
            for k in test_cache[c].keys():
                if k not in result_channels[c][s]['feature_importance']:
                    result_channels[c][s]['feature_importance'][k] = \
                        np.full(len(features[k]), np.nan)
                contrib = ti.predict(self.clf[c].estimators_[k], test_cache[c][k])[2]
                result_channels[c][s]['feature_importance'][k][features[k]] = \
                    np.mean(contrib if self.is_regression else contrib[:, :, 0], axis=0)

        return result_channels

    def _meta_predict(self, y_pred_test_list):

        if self.scheme.clf_meta_args['weighting'] is not None:
            weighting = self.scheme.clf_meta_args['weighting']
            weighted_mean = \
                np.array(y_pred_test_list).swapaxes(0, 1).dot(weighting) / np.sum(weighting)
        else:
            weighted_mean = np.mean(y_pred_test_list, axis=0)

        if self.is_regression:
            return weighted_mean
        else:
            return np.round(weighted_mean)

    def _checks(self):
        """ Make various data checks.

        """
        for i, data in enumerate(self.scheme.data):
            assert np.array_equal(data.labels, self.scheme.data[0].labels), \
                "Labels of all channels must be identical"
        for i, data in enumerate(self.scheme.data):
            assert np.array_equal(data.label_names, self.scheme.data[0].label_names), \
                "Label names of all channels must be identical"
        for i, pipeline in enumerate(self.scheme.channels):
            assert pipeline.clfs.regression == self.scheme.channels[0].clfs.regression, \
                "Channels must be either all regressions or all classifications"

    def _construct_pipe(self):

        """ Construct the processing pipeline (masking, feature selection, classification)

        """
        if hasattr(self.scheme, 'scoring') and self.scheme.scoring is not None:
            self.scoring = getattr(importlib.import_module('sklearn.metrics'), '%s_score' %
                                   self.scheme.scoring)
            self.scoring_name = self.scheme.scoring
        else:
            self.scoring = r2_score if self.is_regression else accuracy_score
            self.scoring_name = 'r2' if self.is_regression else 'accuracy'

        self.masker, self.clf, self.param_grid, self.steps, self.pipeline = \
            [None] * self.n_channels, [None] * self.n_channels, [None] * self.n_channels, \
            [None] * self.n_channels, [None] * self.n_channels
        self.selection = [[] for _ in range(self.n_channels)]

        for c in range(self.n_channels):

            self.param_grid[c] = dict()
            clf = self.scheme.channels[c].clfs.clf
            clf_args = self.scheme.channels[c].clfs.clf_args

            if np.any([isinstance(v, list) for v in self.scheme.masker_args[c].values()]):
                for k, v in self.scheme.masker_args[c].items():
                    if k != 'rois' and isinstance(v, list):
                        self.param_grid[c]['masker__%s' % k] = v

            pipes = [('masker', self.scheme.masker[c](**self.scheme.masker_args[c]))]
            self.masker[c] = pipes[0][1]

            if 'smoothing_fwhm' in self.scheme.masker_args[c] \
                    and isinstance(self.scheme.masker_args[c]['smoothing_fwhm'], list):
                self.param_grid[c]['masker__smoothing_fwhm'] = \
                    self.scheme.masker_args[c]['smoothing_fwhm']

            fs_types = [f.type for f in self.scheme.channels[c].fss] \
                if self.scheme.channels[c].fss is not None else []

            # variance-based feature selection
            if 'variance' in fs_types:
                fs = self.scheme.channels[c].fss[fs_types.index('variance')]
                variance_threshold = MultiRoiVarianceThreshold() if self.scheme.is_multiroi[c] \
                                else VarianceThreshold()
                for k, v in fs.fs_args.items():
                    if isinstance(v, list):
                        self.param_grid[c]['fs_variance__%s' % k] = v
                    else:
                        variance_threshold.set_params(**{k: v})
                pipes.append(('fs_variance', variance_threshold))
                self.selection[c].append(variance_threshold)

            # variance-based feature selection
            if 'variance_percentile' in fs_types:
                fs = self.scheme.channels[c].fss[fs_types.index('variance_percentile')]
                if self.scheme.is_multiroi[c]:
                    variance_threshold = MultiRoiVariancePercentile()
                else:
                    variance_threshold = VariancePercentile()
                for k, v in fs.fs_args.items():
                    if isinstance(v, list):
                        self.param_grid[c]['fs_variance__%s' % k] = v
                    else:
                        variance_threshold.set_params(**{k: v})
                pipes.append(('fs_variance', variance_threshold))
                self.selection[c].append(variance_threshold)

            # F-test-based feature selection
            if 'percentile' in fs_types:
                fs = self.scheme.channels[c].fss[fs_types.index('percentile')]
                f_score = f_regression if self.scheme.channels[c].clfs.regression else f_classif
                percentile = MuliRoiSelectPercentile(score_func=f_score) if \
                    self.scheme.is_multiroi[c] else SelectPercentile(score_func=f_score)
                for k, v in fs.fs_args.items():
                    if isinstance(v, list):
                        self.param_grid[c]['fs_anova__%s' % k] = v
                    else:
                        percentile.set_params(**{k: v})
                pipes.append(('fs_anova', percentile))
                self.selection[c].append(percentile)

            # model-based feature selection
            if 'model' in fs_types or 'nested' in fs_types:
                fs = self.scheme.channels[c].fss[fs_types.index(('nested', 'model')['model' in fs_types])]
                if self.scheme.is_multiroi[c]:
                    select_from = MultiRoiSelectFromModel
                else:
                    select_from = SelectFromModel
                if fs.fs_args['model'] == 'nested':
                    nested_clf = clf(**clf_args)
                    for k, v in clf_args.items():
                        if isinstance(v, list):
                            self.param_grid[c]['fs_model__estimator__%s' % k] = v
                    fs_model = select_from(nested_clf, threshold=fs.fs_args['threshold'])
                else:
                    if 'model_args' not in fs.fs_args:
                        fs.fs_args['model_args'] = dict()
                    for k, v in fs.fs_args['model_args'].items():
                        if isinstance(v, list):
                            self.param_grid[c]['fs_model__estimator__%s' % k] = v
                    fs_model = select_from(fs.fs_args['model'](**fs.fs_args['model_args']),
                                           threshold=fs.fs_args['threshold'])
                if isinstance(fs.fs_args['threshold'], list):
                    self.param_grid[c]['fs_model__threshold'] = fs.fs_args['threshold']

                pipes.append(('fs_model', fs_model))
                self.selection[c].append(fs_model)

            # model-based selection of ROIs
            if 'roi' in fs_types:
                fs = self.scheme.channels[c].fss[fs_types.index('roi')]
                if fs.fs_args['roi_model'] == 'nested':
                    for k, v in clf_args.items():
                        if isinstance(v, list):
                            self.param_grid[c]['fs_roi__estimator__%s' % k] = v
                    if 'base_estimator_args' in clf_args:
                        for k, v in clf_args['base_estimator_args'].items():
                            if isinstance(v, list):
                                self.param_grid[c]['fs_roi__estimator__base_estimator__%s' % k] = v
                    fs_roi = SelectRoisFromModel(clf(**clf_args),
                                                 criterion=fs.fs_args['roi_criterion'])
                else:
                    if 'roi_model_args' not in fs.fs_args:
                        fs.fs_args['roi_model_args'] = dict()
                    for k, v in fs.fs_args['roi_model_args'].items():
                        if isinstance(v, list):
                            self.param_grid[c]['fs_roi__estimator__%s' % k] = v
                    fs_roi = SelectRoisFromModel(fs.fs_args['roi_model'](**fs.fs_args['roi_model_args']),
                                                 criterion=fs.fs_args['roi_criterion'])
                if isinstance(fs.fs_args['roi_criterion'], list):
                    self.param_grid[c]['fs_roi__criterion'] = fs.fs_args['roi_criterion']
                pipes.append(('fs_roi', fs_roi))
                self.selection[c].append(fs_roi)


            # classifier
            # if 'class_weight' in signature(clf.__init__).parameters.keys():
            #     clf_args.update(class_weight='balanced')
            if self.scheme.is_multiroi[c]:
                be_dict = {'base_estimator': clf(**clf_args)}
                multiroi_args = {**be_dict, **self.scheme.clf_multiroi_args[c]}
                self.clf[c] = self.scheme.clf_multiroi[c](**multiroi_args)
                if np.any([isinstance(v, list) for v in clf_args.values()]):
                    for k, v in clf_args.items():
                        if isinstance(v, list):
                            self.param_grid[c]['clf__base_estimator__%s' % k] = v
            else:
                if self.n_jobs_folds is not None and self.searchlight:
                    clf_args.update(n_jobs=self.n_jobs_folds)
                if 'verbose' in signature(clf).parameters.keys():
                    clf_args.update(verbose=max(0, self.verbose-2))
                self.clf[c] = clf(**clf_args)
                if np.any([isinstance(v, list) for v in clf_args.values()]):
                    for k, v in clf_args.items():
                        if isinstance(v, list):
                            self.param_grid[c]['clf__%s' % k] = v
            pipes.append(('clf', self.clf[c]))

            if not len(self.param_grid[c]):
                self.param_grid[c] = None

            # create final pipeline and parameter grid
            self.pipeline[c] = Pipeline(pipes)
            self.steps[c] = self.pipeline[c].steps
            if self.param_grid[c] is not None:
                self.pipeline[c] = GridSearchCV(self.pipeline[c], param_grid=self.param_grid[c],
                                                verbose=max(0, self.verbose-3), refit=True,
                                                scoring='accuracy')