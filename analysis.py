import importlib
import pickle
import os
import time
import timeit
import warnings
from inspect import signature
from numbers import Number

import multiprocessing_on_dill as multiprocessing
import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.stats import sem
from sklearn.discriminant_analysis import _cov as sklearn_cov
from sklearn.metrics import accuracy_score, r2_score, explained_variance_score, \
    confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC, SVR
from tabulate import tabulate
from treeinterpreter import treeinterpreter as ti

from .cv import DummyCV, ExhaustiveLeave2Out
from .grid_search import MultiModalGridSearchCV, GridSearchCVLabels
from .grid_search_separate import MultiModalSeparateGridSearchCV
from .metrics import bacc_p, bacc_ppi
from .pipeline import MultiModalFeatureUnion
from .util.various import elapsed_time
from .descriptor import Struct
from scipy.stats import pearsonr
from eli5.sklearn.permutation_importance import PermutationImportance
from decog.estimators.permutation_importance_new import PermutationImportanceMultiModal
from collections import OrderedDict
from .estimators.metaclf import MetaClassifier

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

        self.searchlight = self.scheme.searchlight

        self.n_folds = None
        self.cv = None
        self.n_seeds = None
        self.n_samples = None
        self.seed_list = None
        self.is_regression = None
        self.y_true = None
        self.label_names = None
        self.scoring = None

        # self.masker = None
        # self.clf = None
        # self.pipeline = None
        # self.steps = None
        # self.param_grid = None

        self.verbose = None
        self.detailed_save = None

    def run(self, n_jobs_folds=1, n_jobs_grid=1, verbose=2, detailed_save=True, cv_pre=None):

        """ Start the analysis.

        Parameters
        ----------
        n_jobs_folds : int
            Number of processors for cross-validation
        verbose : int
            verbosity level
        detailed_save : boolean
            compute and save additional information

        Returns
        -------
        dict

        """

        self.verbose = verbose
        self.detailed_save = detailed_save
        self.n_jobs_folds = n_jobs_folds
        self.n_jobs_grid = n_jobs_grid
        self.is_regression = self.scheme.regression
        self.y_true = np.array(self.scheme.labels)
        self.label_names = self.scheme.label_names
        self.has_time = self.scheme.has_time

        self.seed_lists = self.scheme.seed_list
        self.n_seeds = self.scheme.n_seeds
        self.n_samples = len(self.y_true)

        if cv_pre is None:
            if not self.searchlight:
                if not hasattr(self.scheme, 'cv') or self.scheme.cv is None:
                    self.cv = LeaveOneOut()
                    # from sklearn.cross_validation import KFold
                    # self.cv_outer = KFold(self.n_samples)
                else:
                    self.cv = self.scheme.cv
            else:  # Nilearn searchlight algorihm uses an internal CV, so here we use a dummy CV
                self.cv = DummyCV(self.n_samples)

        self._construct_pipe()
        if isinstance(self.clf, SVC):
            print('\n\nC: %s / %s' % (np.log2(self.clf.C), self.clf.C))

        kwargs = dict()
        if self.has_time:
            func_fold = self._fold_time
            if isinstance(self.data, str):
                kwargs['X'] = pickle.load(open(self.data, 'rb'))
            else:
                kwargs['X'] = self.data
        else:
            func_fold = self._fold

        if hasattr(self, 'preproc'):
            if self.is_multimodal:
                for k, X in self.data.items():
                    if isinstance(X, np.ndarray):
                        valid = ~np.isnan(X[:, 0])
                        self.data[k][valid] = self.preproc.fit_transform(X[valid])
                    else:
                        raise ValueError('Preprocessing only implemented for data of form numpy.ndarray')
            else:
                if isinstance(self.data, np.ndarray):
                    self.data = self.preproc.fit_transform(self.data)
                elif self.has_time and isinstance(kwargs['X'], np.ndarray):
                    kwargs['X'] = np.moveaxis([self.preproc.fit_transform(kwargs['X'][:, :, t]) for t in range(kwargs['X'].shape[2])], 0, 2)
                else:
                    raise ValueError('Preprocessing only implemented for data of form numpy.ndarray')

        if cv_pre is not None:
            f, train_index, test_index, self.n_folds = cv_pre
            result = func_fold((f, train_index, test_index, kwargs))[0]
        else:
            cv_data = self.data[list(self.data.keys())[0]] if self.is_multimodal else self.data
            self.n_folds = self.cv.get_n_splits(X=cv_data, y=self.y_true)
            if n_jobs_folds == 1:
                results_folds = []
                for f, (train_index, test_index) in enumerate(self.cv.split(cv_data, self.y_true)):
                    results_folds.append(func_fold((f, train_index, test_index, kwargs)))
            else:
                pool = multiprocessing.Pool(None if n_jobs_folds == -1 else n_jobs_folds)
                results_folds = pool.map(func_fold,
                                   [(f, train_index.copy(), test_index.copy(), kwargs)
                                    for f, (train_index, test_index) in enumerate(self.cv.split(cv_data, self.y_true))])
                pool.close()
            if self.has_time:
                result = dict()
                # Average across folds
                result['result'] = np.mean(results_folds, axis=0)
                result['folds'] = results_folds
            elif not self.searchlight:
                result = self._post_processing(results_folds)
                if verbose > -2:
                    self._print_result(result)
            else:
                result = {'seed%g' % s: results_folds[0][0][0][s]['searchlight']
                          for s in range(self.n_seeds[0])}
                print('Searchlight finished!')

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

        if self.verbose > 1:
            t0 = time.time()

        f, train_indices, test_indices, kwargs = params
        n_test = len(test_indices)

        ### Prepare data structures ###
        result = [None] * self.n_seeds
        for s in range(self.n_seeds):
            result[s] = dict()
        ###

        if self.is_multimodal:
            data_train = {k: [v[i] for i in train_indices] for k, v in self.data.items()}
            data_test = {k: [v[i] for i in test_indices] for k, v in self.data.items()}
        else:
            data_train = [self.data[i] for i in train_indices]
            data_test = [self.data[i] for i in test_indices]

        # if isinstance(self.data, dict):
        #     X_train = {k: [v[i] for i in train_indices] for k, v in self.data.items()}
        #     data_test = {k: [v[i] for i in test_indices] for k, v in self.data.items()}
        # else:
        #     X_train = [self.data[i] for i in train_indices]
        #     data_test = [self.data[i] for i in test_indices]
        labels_train = [self.y_true[i] for i in train_indices]

        test_cache, train_cache = None, None

        for s, seed_ in enumerate(self.seed_lists):

            if self.searchlight:
                print('[%s] Starting searchlight analysis' % time.strftime("%d.%m %H:%M:%S"))
            elif self.verbose > 1:
                print('[%s] Fold %g / %g Seed %g / %g' %
                      (time.strftime("%d.%m %H:%M:%S"),
                       f + 1, self.n_folds,
                       s + 1, self.n_seeds))
                if self.name and self.verbose > 2:
                    print('\tof << %s >>' % self.name)

            ### Set seeds ###
            if hasattr(self.clf, 'random_state'):
                self.clf.random_state = seed_

            ### (1 / 2) If no GridSearchCV or feature selection (with varying seeds) is used, we
            ### can prefit the preprocessing steps
            if not self.is_grid and train_cache is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    # t0 = time.time()
                    # print('Start [0]')
                    train_cache = Pipeline(self.steps[:-1]).fit_transform(data_train, labels_train)
                    # print('[0] Elapsed time: %.2f secs' % (time.time() - t0))
            ###

            if hasattr(self.scheme, 'prefit') and self.scheme.prefit is not None:
                if hasattr(self.scheme, 'prefit_exceptions') and self.scheme.prefit_exceptions is not None:
                    pes = dict()
                    for k in self.scheme.prefit_exceptions:
                        pes.update({k: getattr(self.clf, k)})
                self.clf, self.pipeline, self.steps = \
                    pickle.load(open(os.path.join(self.scheme.prefit, '%0*g.pkl' % (len(str(self.n_folds)), f + 1)), 'rb'))
                if hasattr(self.scheme, 'prefit_exceptions') and self.scheme.prefit_exceptions is not None:
                    for k, v in pes.items():
                        setattr(self.clf, k, v)

            else:
                ### Fit training data ###
                if self.is_grid:
                    self.pipeline.cv = self.scheme.cv_grid

                    # otherwise GridSearchCV won't be silent
                    ind_clf = [el[0] for el in self.pipeline.estimator.steps].index('clf')
                    if not isinstance(self.steps[ind_clf][1], (SVC, SVR, LinearSVC)):
                        pass
                        # self.steps[ind_clf][1].verbose = self.pipeline.verbose
                    else:
                        self.steps[ind_clf][1].verbose = False

                    # print(self.pipeline)
                    # print(labels_train)
                    # print(data_train)
                    self.pipeline.fit(X=data_train, y=labels_train)


                    for step in range(len(self.steps)):
                        self.steps[step] = self.pipeline.best_estimator_.steps[step]

                    self.clf = self.pipeline.best_estimator_._final_estimator

                    Pipeline(self.pipeline.estimator.steps[:-1]).transform(data_train)
                else:
                    if train_cache is not None:
                        self.clf.fit(train_cache, labels_train)
                    else:
                        self.pipeline.fit(data_train, labels_train)
                ###

            if hasattr(self.scheme, 'postfit') and self.scheme.postfit is not None:
                path_postfit = os.path.join(self.scheme.postfit, '%0*g.pkl' % (len(str(self.n_folds)), f + 1))
                pickle.dump((self.clf, self.pipeline, self.steps), open(path_postfit, 'wb'))
                print('\tSaved prefit file to %s' % path_postfit)

            ### Transform test data ###
            if test_cache is None or \
                ('fs_model' in dict(self.steps) and seed_ is not None):
                test_cache = Pipeline(self.steps[:-1]).transform(data_test)
                ###

            ### Prediction ###
            if hasattr(self.scheme, 'clf_fold_attr') and self.scheme.clf_fold_attr is not None:
                for k, v in self.scheme.clf_fold_attr.items():
                    setattr(self.clf, k, v[f])
            predictions = self.clf.predict(test_cache)
            ###

            ### Collect GridSearchCV results ###

            if self.is_grid:
                if isinstance(self.pipeline, MultiModalSeparateGridSearchCV):
                    # result[s]['grid'] = sum([['[%s] %s' % (m, str(v)) for v in r['params']] for m, r in self.pipeline.cv_results_.items()], [])
                    result[s]['grid'] = {m: [str(v) for v in r['params']] for m, r in self.pipeline.cv_results_.items()}
                    result[s]['grid_scores'] = {m: r['mean_test_score'] for m, r in self.pipeline.cv_results_.items()}

                    for m, r in self.pipeline.cv_results_.items():
                        params_best = r['params'][r['rank_test_score'].argmin()]
                        for k, v in params_best.items():
                            result[s]['param__%s__%s' % (m, k)] = v

                    if self.verbose > 2:
                        for m in self.pipeline.cv_results_.keys():
                            for grid, score in zip(result[s]['grid'][m],
                                                   result[s]['grid_scores'][m]):
                                print('[%s] %s: %.5f' % (m, grid, score))
                else:
                    result[s]['grid'] = [str(v) for v in self.pipeline.cv_results_['params']]
                    result[s]['grid_scores'] = self.pipeline.cv_results_['mean_test_score']

                    params_best = self.pipeline.cv_results_['params'][self.pipeline.cv_results_['rank_test_score'].argmin()]
                    for k, v in params_best.items():
                        result[s]['param__%s' % k] = v

                    if self.verbose > 2:
                        for grid, score in zip(result[s]['grid'],
                                               result[s]['grid_scores']):
                            print('%s: %.5f' % (grid, score))
            ###


            if isinstance(self.clf, MetaClassifier) and hasattr(self.clf, 'weighting') and self.clf.weighting is not None:
                result[s]['weighting'] = self.clf.weighting

            ### Collect classifier variables ###
            if self.scheme.compute_permutation_importance:
                # if self.verbose > 0:
                #     print('Evaluating permutation feature importance..')
                PI = PermutationImportanceMultiModal if self.is_multimodal else PermutationImportance
                labels_test = [self.y_true[i] for i in test_indices]
                # perm = PI(self.clf, scoring=self.scoring_name).fit(data_test, labels_test)
                niter = self.scheme.permutation_importance_niter if hasattr(self.scheme, 'permutation_importance_niter') else 10
                perm = PI(self.pipeline, scoring=self.scoring_name, n_iter=niter).fit(data_test, labels_test)
                result[s]['permutation_importance'] = perm.feature_importances_
            if self.is_multimodal:
                for k in self.channel.keys():
                    if self.channel[k].clfs.collect_variables is not None:
                        for v in self.channel[k].clfs.collect_variables:
                            result[s]['clfvar_%s_%s' % (k, v)] = getattr(self.clf.base_estimators[k], v)
            elif self.channel.clfs.collect_variables is not None:
                for v in self.channel.clfs.collect_variables:
                    result[s]['clfvar_%s' % v] = getattr(self.clf, v)

            # if not self.is_multimodal and self.scheme.feature_importances:
            #     if isinstance(self.clf, (RandomForestRegressor, RandomForestClassifier)):
            #         result[s]['feature_importances'] = self._forest_importances(X, test_cache, result, c, s)
            #     elif isinstance(self.clf, SVC):
            #         result[s]['feature_importances'] = self._svc_importances(result, s)
            ###

            ### Store predictions ###
            if not self.searchlight:
                result[s]['y_pred'] = predictions.astype(float)
                if hasattr(self.clf, 'votes_pooled'):
                    result[s]['votes_pooled'] = self.clf.votes_pooled
                if hasattr(self.clf, 'votes'):
                    result[s]['votes'] = dict()
                    for i, k in enumerate(self.clf.estimators_.keys()):
                        result[s]['votes'][k] = np.atleast_2d(self.clf.votes)[i, :]
            else:
                result[s]['searchlight'] = predictions
            ###

        if self.verbose > 1:
            print('\t[Fold %g / %g Seed %g / %g Finished in %s]' %
                      (f + 1, self.n_folds,
                       s + 1, self.n_seeds,
                       elapsed_time(time.time() - t0)))

        return result


    def _fold_time(self, params):

        f, train_indices, test_indices, kwargs = params
        X = kwargs['X']

        if self.verbose > 1:
            tic = timeit.default_timer()
            print('[%s] Permutation %g / %g' %
                  (time.strftime("%d.%m %H:%M:%S"), f + 1, self.n_folds))

        n_time = X.shape[2]
        n_sensors = X.shape[1]
        n1, n2 = self.cv.ind_pseudo_test.shape[:2] if len(self.cv.ind_pseudo_test) else (self.cv.n_classes, self.cv.n_classes)

        if hasattr(self.scheme, 'remove_pattern_mean') and self.scheme.remove_pattern_mean:
            X = X - X.mean(axis=0)


        if hasattr(self.scheme, 'zscoring') and self.scheme.zscoring:
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            for time_point in range(n_time):
                for c1 in range(n1 - 1 + (n2 == 1)):
                    for c2 in range(min(c1 + 1, n2 - 1), n2):
                        trainind = np.sort(sum([train_indices[i].tolist() for i in self.scheme.cv.ind_pseudo_train[c1, c2]], []))
                        testind = np.sort(sum([test_indices[i].tolist() for i in self.scheme.cv.ind_pseudo_test[c1, c2]], []))
                        X[trainind, :, time_point] = sc.fit_transform(X[trainind, :, time_point])
                        X[testind, :, time_point] = sc.transform(X[testind, :, time_point])
                        
                        # trainind1 = trainind[self.scheme.cv.labels[trainind]==c1]
                        # X[trainind1, :, time_point] = sc.fit_transform(X[trainind1, :, time_point])
                        # testind1 = testind[self.scheme.cv.labels[testind]==c1]
                        # X[testind1, :, time_point] = sc.transform(X[testind1, :, time_point])
                        # trainind2 = trainind[self.scheme.cv.labels[trainind]==c2]
                        # X[trainind2, :, time_point] = sc.fit_transform(X[trainind2, :, time_point])
                        # testind2 = testind[self.scheme.cv.labels[testind]==c2]
                        # X[testind2, :, time_point] = sc.transform(X[testind2, :, time_point])
            print('[%s] Z-Scoring finished!' % time.strftime("%d.%m %H:%M:%S"))

        Xpseudo_train = np.full((len(train_indices), n_sensors,
                                 n_time), np.nan)
        Xpseudo_test = np.full((len(test_indices), n_sensors, n_time), np.nan)

        result = np.full((n1, n2, n_time), np.nan)

        for i, order_ in enumerate(train_indices):
            Xpseudo_train[i, :, :] = np.mean(X[order_, :, :], axis=0)
        for i, order_ in enumerate(test_indices):
            Xpseudo_test[i, :, :] = np.mean(X[order_, :, :], axis=0)

        # Xbaseline = X[train_indices[0]][:, :, self.scheme.baseline]
        # for i, order_ in enumerate(train_indices[1:]):
        #     Xbaseline = np.vstack((Xbaseline, X[order_][:, :, self.scheme.baseline]))
        #
        # if self.clf.__class__ == decog.estimators.distance.PyRiemann:
        #

        sigma_method = False
        univ = False
        if hasattr(self.clf, 'sigma_method'):
            sigma_method = self.clf.sigma_method
            if '_univ' in sigma_method:
                self.clf.sigma_method = self.clf.sigma_method.replace('_univ', '')
                sigma_method = self.clf.sigma_method
                univ = True
        if hasattr(self.clf, 'estimator') and hasattr(self.clf.estimator, 'sigma_method'):
            sigma_method = self.clf.estimator.sigma_method
            if '_univ' in sigma_method:
                self.clf.estimator.sigma_method = self.clf.estimator.sigma_method.replace('_univ', '')
                sigma_method = self.clf.estimator.sigma_method
                univ = True
        if hasattr(self.scheme, 'sigma_method'):  # takes priority
            sigma_method = self.scheme.sigma_method
            if '_univ' in sigma_method:
                self.scheme.sigma_method = self.scheme.sigma_method.replace('_univ', '')
                sigma_method = self.scheme.sigma_method
                univ = True

        shrinkage = None
        if hasattr(self.clf, 'shrinkage'):
            shrinkage = self.clf.shrinkage
        if hasattr(self.scheme, 'shrinkage'):  # takes priority
            shrinkage = self.scheme.shrinkage


        sigma_data = Xpseudo_train if len(train_indices) else Xpseudo_test
        if sigma_method and sigma_method in ['timepoint', 'timepoint2', 'full', 'full2', 'signal', 'signal2', 'timepoint_window2']:
            if len(self.cv.labels_pseudo_train):
                sigma_conditions = self.cv.labels_pseudo_train[0, :, int(self.cv.labels_pseudo_train.shape[2]/2):].flatten()
            elif self.cv.labels_pseudo_test.shape[1] == 1:
                sigma_conditions = self.cv.labels_pseudo_test.flatten()
            else:
                sigma_conditions = self.cv.labels_pseudo_test[0, :, int(self.cv.labels_pseudo_test.shape[2]/2):].flatten()
            sigma_conditions_uniq = np.unique(sigma_conditions)
            if sigma_method in ['timepoint', 'full', 'signal']:
                for c in np.unique(sigma_conditions):
                    sigma_data[sigma_conditions==c] -= sigma_data[sigma_conditions==c].mean(axis=0)

        if hasattr(self.clf, 'sigma') or hasattr(self.scheme, 'sigma') or (sigma_method and hasattr(self.clf, 'estimator')):
            if sigma_method and 'timepoint' in sigma_method:
                if sigma_method == 'timepoint':
                    sigma = np.array([sklearn_cov(sigma_data[:, :, t], shrinkage=shrinkage) for t in range(sigma_data.shape[2])])
                elif sigma_method == 'timepoint2':
                    sigma_ = np.empty((sigma_data.shape[2], len(sigma_conditions_uniq), n_sensors, n_sensors))
                    for i, c in enumerate(sigma_conditions_uniq):
                        sigma_[:, i] = [sklearn_cov(sigma_data[sigma_conditions==c, :, t], shrinkage=shrinkage) for t in range(sigma_data.shape[2])]
                    sigma = sigma_.mean(axis=1)
                elif sigma_method == 'timepoint_window2':
                    ws = 50 if '1ms' in self.scheme.name else 5
                    sigma_ = np.empty((sigma_data.shape[2], len(sigma_conditions_uniq), n_sensors, n_sensors))
                    for i, c in enumerate(sigma_conditions_uniq):
                        sigma_[:, i] = [sklearn_cov(sigma_data[sigma_conditions==c, :, t], shrinkage=shrinkage) for t in range(sigma_data.shape[2])]
                    sigma_ = sigma_.mean(axis=1)
                    sigma = np.array([sigma_[min(max(0, t-ws), sigma_data.shape[2]-2*ws):max(min(sigma_data.shape[2], t+ws+1), 2*ws)].mean(axis=0) for t in range(sigma_data.shape[2])])
                if univ:
                    sigma = np.array([s*np.eye(s.shape[0]) for s in sigma])
            else:
                if not sigma_method or sigma_method == 'baseline1':
                    self.clf.sigma = sklearn_cov(sigma_data[:, :, self.scheme.baseline].swapaxes(0, 1).reshape(-1, Xpseudo_train.shape[1]),
                                                 shrinkage=shrinkage)
                elif sigma_method == 'baseline2':
                    self.clf.sigma = np.array([sklearn_cov(sigma_data[:, :, t], shrinkage=shrinkage) for t in self.scheme.baseline]).mean(axis=0)
                elif sigma_method == 'baseline3':
                    self.clf.sigma = sklearn_cov(sigma_data[:, :, self.scheme.baseline].mean(axis=2), shrinkage=shrinkage)
                elif sigma_method == 'full':
                    self.clf.sigma = np.array([sklearn_cov(sigma_data[:, :, t], shrinkage=shrinkage) for t in range(sigma_data.shape[2])]).mean(axis=0)
                elif sigma_method == 'full2':
                    sigma_ = np.empty((len(sigma_conditions_uniq), n_sensors, n_sensors))
                    for i, c in enumerate(sigma_conditions_uniq):
                        sigma_[i] = np.mean([sklearn_cov(sigma_data[sigma_conditions==c, :, t], shrinkage=shrinkage) for t in range(sigma_data.shape[2])], axis=0)
                    self.clf.sigma = sigma_.mean(axis=0)
                    # self.clf.sigma = np.array([[sklearn_cov(sigma_data[sigma_conditions==c, :, t], shrinkage=shrinkage) for t in range(sigma_data.shape[2])] for c in np.unique(sigma_conditions)]).mean(axis=(0, 1))
                elif sigma_method == 'signal':
                    window = np.arange(150, 450) if '1ms' in self.scheme.name else np.arange(15, 45)
                    self.clf.sigma = sklearn_cov(sigma_data[:, :, window].mean(axis=2), shrinkage=shrinkage)
                elif sigma_method == 'signal2':
                    window = np.arange(150, 450) if '1ms' in self.scheme.name else np.arange(15, 45)
                    sigma_ = np.empty((len(sigma_conditions_uniq), n_sensors, n_sensors))
                    for i, c in enumerate(sigma_conditions_uniq):
                        sigma_[i] = np.mean([sklearn_cov(sigma_data[sigma_conditions==c, :, t], shrinkage=shrinkage) for t in window], axis=0)
                    self.clf.sigma = sigma_.mean(axis=0)
                elif sigma_method == 'identity':
                    self.clf.sigma = np.eye(n_sensors)
                # self.clf.sigma = np.cov(Xpseudo_train[:, :, self.scheme.baseline].swapaxes(0, 1).reshape(-1, Xpseudo_train.shape[1]).T)
                # self.clf.sigma = np.cov(Xbaseline[:, :, self.scheme.baseline].swapaxes(0, 1).reshape(-1, Xbaseline.shape[1]).T)
                if univ:
                    self.clf.sigma = self.clf.sigma*np.eye(self.clf.sigma.shape[0])

                if hasattr(self.clf, 'estimator'):
                    self.clf.estimator.sigma = self.clf.sigma

        if hasattr(self.clf, 'fractional_sigma') or hasattr(self.scheme, 'fractional_sigma'):
            if sigma_method and 'timepoint' in sigma_method:
                fractional_sigma = np.array([fractional_matrix_power(sigma[t], -0.5) for t in range(sigma_data.shape[2])])
            else:
                self.clf.fractional_sigma = fractional_matrix_power(self.clf.sigma, -0.5)
        if hasattr(self.clf, 'inv_sigma') or hasattr(self.scheme, 'inv_sigma'):
            if sigma_method and 'timepoint' in sigma_method:
                inv_sigma = np.array([np.linalg.inv(sigma[t]) for t in range(sigma_data.shape[2])])
            else:
                self.clf.inv_sigma = np.linalg.inv(self.clf.sigma)


        if hasattr(self.scheme, 'whiten') and self.scheme.whiten:
            if (self.verbose > 2) and hasattr(self.scheme, 'sigma_method'):
                print("Whitening the data with sigma method '%s'" % self.scheme.sigma_method)
            if hasattr(self.scheme, 'sigma_method') and 'timepoint' in self.scheme.sigma_method:
                if len(Xpseudo_train):
                    for t in range(Xpseudo_train.shape[2]):
                        Xpseudo_train[:, :, t] = Xpseudo_train[:, :, t] @ fractional_sigma[t]
                if len(Xpseudo_test):
                    for t in range(Xpseudo_test.shape[2]):
                        Xpseudo_test[:, :, t] = Xpseudo_test[:, :, t] @ fractional_sigma[t]
            else:
                if len(Xpseudo_train):
                    Xpseudo_train = (Xpseudo_train.swapaxes(1, 2) @ self.clf.fractional_sigma).swapaxes(1, 2)
                if len(Xpseudo_test):
                    Xpseudo_test = (Xpseudo_test.swapaxes(1, 2) @ self.clf.fractional_sigma).swapaxes(1, 2)

        for time_point in range(n_time):
            if sigma_method and 'timepoint' in sigma_method:
                if hasattr(self.clf, 'sigma'):
                    self.clf.sigma = sigma[time_point]
                if hasattr(self.clf, 'fractional_sigma'):
                    self.clf.fractional_sigma = fractional_sigma[time_point]
                if hasattr(self.clf, 'inv_sigma'):
                    self.clf.inv_sigma = inv_sigma[time_point]
            for c1 in range(n1-1+(n2==1)):
                for c2 in range(min(c1 + 1, n2-1), n2):
                        if len(train_indices):
                            data_train = Xpseudo_train[self.cv.ind_pseudo_train[c1, c2], :, time_point]
                            self.clf.fit(data_train, self.cv.labels_pseudo_train[c1, c2])

                        data_test = Xpseudo_test[self.cv.ind_pseudo_test[c1, c2], :, time_point]

                        if len(train_indices):
                            if hasattr(self.scheme, 'remove_pattern_mean_cv') and self.scheme.remove_pattern_mean_cv:
                                data_test -= data_train.mean(axis=0)
                                if not c1:
                                    print('remove_pattern_mean_cv performed')
                        else:
                            cvind = np.setdiff1d(np.arange(Xpseudo_test.shape[0]), self.cv.ind_pseudo_test[c1, c2])
                            data_test -= Xpseudo_test[cvind, :, time_point].mean(axis=0)
                            if not c1:
                                print('remove_pattern_mean_cv performed')


                        if self.clf._estimator_type == 'distance':
                            dissimilarity = self.clf.predict(data_test, y=self.cv.labels_pseudo_test[c1, c2])
                        elif self.clf._estimator_type == 'DV':
                            DV = self.clf.predict(data_test)
                            dissimilarity = np.mean(np.abs(DV) * ((self.clf.estimator.classes_[(DV > 0).astype(int)] == self.cv.labels_pseudo_test[c1, c2]) - 0.5))
                        elif self.clf._estimator_type == 'AUC':
                            DV = self.clf.predict(data_test)
                            dissimilarity = roc_auc_score(self.cv.labels_pseudo_test[c1, c2], DV)
                        else:
                            predictions = self.clf.predict(data_test)
                            dissimilarity = np.mean(predictions == self.cv.labels_pseudo_test[c1, c2]) - 0.5
                        result[c1, c2, time_point] = np.mean(dissimilarity)

        if self.verbose > 1:
            print('Permutation time: %.2f secs' % (timeit.default_timer() - tic))

        return result

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
        result_seeds = [None] * self.n_seeds
        if self.is_grid:
            grid_params = [k for k in results[0][0].keys() if k.startswith('param__')]
            if isinstance(self.pipeline, MultiModalSeparateGridSearchCV):
                mod = np.unique([p.split('__')[1] for p in grid_params])
                grid_params = {m: [p for p in grid_params if p.startswith('param__%s' % m)] for m in mod}
        for s in range(self.n_seeds):
            result_seeds[s] = dict(
                y_pred=np.full(self.n_samples, np.nan),
            )
            if hasattr(self.clf, 'votes'):
                result_seeds[s]['votes'] = \
                    np.full((len(self.scheme.channels.masker.masker_args['rois']), self.n_samples), np.nan)
            if hasattr(self.clf, 'votes_pooled'):
                result_seeds[s]['votes_pooled'] = np.full(self.n_samples, np.nan)
            if self.is_grid:
                if isinstance(self.pipeline, MultiModalSeparateGridSearchCV):
                    result_seeds[s]['grid_scores'] = {m: [None] * self.n_folds for m in mod}
                    for m, grid_params_ in grid_params.items():
                        result_seeds[s][m] = dict()
                        for grid_param in grid_params_:
                            result_seeds[s][m][grid_param] = [None] * self.n_folds
                else:
                    result_seeds[s]['grid_scores'] = [None] * self.n_folds
                    for grid_param in grid_params:
                        result_seeds[s][grid_param] = [None] * self.n_folds

        cv_data = self.data[list(self.data.keys())[0]] if self.is_multimodal else self.data
        for f, (train_index, test_index) in enumerate(self.cv.split(cv_data, self.y_true)):
            for s in range(self.n_seeds):
                result_seeds[s]['y_pred'][test_index] = results[f][s]['y_pred']
                if hasattr(self.clf, 'votes'):
                    result_seeds[s]['votes'][:, test_index] = results[f][s]['votes']
                if hasattr(self.clf, 'votes_pooled'):
                    result_seeds[s]['votes_pooled'][test_index] = results[f][s]['votes_pooled']
                if self.is_grid:
                    if isinstance(self.pipeline, MultiModalSeparateGridSearchCV):
                        for m in mod:
                            result_seeds[s]['grid_scores'][m][f] = results[f][s]['grid_scores'][m]
                            for grid_param in grid_params[m]:
                                result_seeds[s][m][grid_param][f] = results[f][s][grid_param]
                    else:
                        result_seeds[s]['grid_scores'][f] = results[f][s]['grid_scores']
                        for grid_param in grid_params:
                            result_seeds[s][grid_param][f] = results[f][s][grid_param]

        result = dict()
        result = self._assess_performance(result_seeds, container=result)

        if 'weighting' in results[0][0]:
            result['weighting'] = {m: np.array([[results[f][s]['weighting'][m] for f in range(self.n_folds)] for s in range(self.n_seeds)]).mean(axis=0).tolist() for m in results[0][0]['weighting'].keys()}

        if hasattr(self.clf, 'votes'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result['votes'] = \
                    np.nanmean([result_seeds[s]['votes']
                                for s in range(self.n_seeds)], axis=0).tolist()
        if hasattr(self.clf, 'votes_pooled'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result['votes_pooled'] = \
                    np.nanmean([result_seeds[s]['votes_pooled']
                             for s in range(self.n_seeds)], axis=0).tolist()

        if self.scheme.compute_permutation_importance:
            if self.is_multimodal:
                result['permutation_importance'], result['permutation_importance_sem'] = dict(), dict()
                for k in results[0][0]['permutation_importance'].keys():
                    result['permutation_importance'][k] = np.mean([[results[f][s]['permutation_importance'][k] for f in range(self.n_folds)] for s in range(self.n_seeds)], axis=0).tolist()
            else:
                result['permutation_importance'] = np.mean([[results[f][s]['permutation_importance'] for f in range(self.n_folds)] for s in range(self.n_seeds)], axis=0).tolist()

        if self.is_multimodal:
            for k in self.channel.keys():
                if self.channel[k].clfs.collect_variables is not None:
                    for v in self.channel[k].clfs.collect_variables:
                        result['clfvar_%s_%s' % (k, v)] = np.array([[results[f][s]['clfvar_%s_%s' % (k, v)] for f in range(self.n_folds)] for s in range(self.n_seeds)]).mean(axis=0).tolist()
        elif self.channel.clfs.collect_variables is not None:
            for v in self.channel.clfs.collect_variables:
                result['clfvar_%s' % v] = np.array([[results[f][s]['clfvar_%s' % v] for f in range(self.n_folds)] for s in range(self.n_seeds)]).mean(axis=0).tolist()
                # result['clfvar_%s' % v] = np.array([[results[f][s]['clfvar_%s' % v] for f in range(self.n_folds)] for s in range(self.n_seeds)]).mean(axis=(0,1))
                # for f in range(len(self.cv)):
                #     for s in range(self.n_seeds):
                #         if 'feature_importance' not in result_seeds[s]:
                #             result_seeds[s]['feature_importance'] = \
                #                 np.full((self.n_folds,
                #                          len(results[f][s]['feature_importance'])),
                #                         np.nan)
                #         result_seeds[s]['feature_importance'][f, :] = \
                #             results[f][s]['feature_importance']
                # with warnings.catch_warnings():  # catch stupid behavior of nanmean
                #     warnings.simplefilter("ignore", RuntimeWarning)
                #     result['feature_importance'] = \
                #         np.nanmean([result_seeds[s]['feature_importance']
                #                     for s in range(self.n_seeds)], axis=(0, 1)).tolist()

        if self.is_grid:
            if isinstance(self.pipeline, MultiModalSeparateGridSearchCV):
                grid_scores_mean = \
                    {m: np.mean([r for s in range(self.n_seeds)], axis=(0, 1)).tolist() for m, r in result_seeds[s]['grid_scores'].items()}
                grid_scores_ste = \
                    {m: (np.std([r for s in range(self.n_seeds)], axis=(0, 1))/np.sqrt(self.n_folds)).tolist() for m, r in result_seeds[s]['grid_scores'].items()}
                result['grid_scores'] = \
                    {m: [(param, grid_scores_mean[m][i], grid_scores_ste[m][i]) for i, param in enumerate(r)] for m, r in results[0][0]['grid'].items() if m in grid_scores_mean}
                for m, grid_params_ in grid_params.items():
                    for grid_param in grid_params_:
                        if isinstance(result_seeds[0][m][grid_param][0], Number):
                            result[grid_param] = np.mean([result_seeds[s][m][grid_param] for s in range(self.n_seeds)], axis=0).tolist()
                        else:
                            result[grid_param] = [result_seeds[s][m][grid_param] for s in range(self.n_seeds)]
            else:
                grid_scores_mean = \
                    np.mean([result_seeds[s]['grid_scores']
                             for s in range(self.n_seeds)], axis=(0, 1)).tolist()
                grid_scores_ste = \
                    (np.std(np.mean([result_seeds[s]['grid_scores']
                                     for s in range(self.n_seeds)], axis=1), axis=0) /
                     self.n_folds).tolist()
                result['grid_scores'] = \
                    [(param, grid_scores_mean[i], grid_scores_ste[i])
                     for i, param in
                     # enumerate([scr.parameters for scr in self.pipeline.grid_scores_])]
                     enumerate(results[0][0]['grid'])]
                for grid_param in grid_params:
                    if isinstance(result_seeds[0][grid_param][0], Number):
                        result[grid_param] = np.mean([result_seeds[s][grid_param]
                                                      for s in range(self.n_seeds)], axis=0).tolist()
                    else:
                        result[grid_param] = [result_seeds[s][grid_param] for s in range(self.n_seeds)]


        return result

    def _print_result(self, result):

        """

        Parameters
        ----------
        result : dict
            Dictionary containing the results
        """

        if self.verbose > -2:
            print("\n**************  %s  **************" % time.strftime("%Y/%m/%d %H:%M:%S"))
            print('Scheme: %s' % self.scheme.name)
            print("%s: %.5f +- %.5f %s" %
                  (str(self.scoring).split(' ')[1], result['scoring'],
                   result['scoring_ste'],
                  ['%.5f' % acc for acc in result['scoring_seed']]))
            if self.is_regression:
                print('Explained variance: %.4f%%' %
                      (100 * result['explained_variance']))
                r, p = pearsonr(self.scheme.labels, result['predictions'])
                print('r = %.5f (p = %.7f))' % (r, p))
            else:
                print('Balanced CI: [%.5f %.5f] balanced p = %.5f (%.2E)' %
                      (100*result['CI_balanced'][0], 100*result['CI_balanced'][1],
                       result['p_balanced'], result['p_balanced']))
                # if 'classification_report' in result.keys():
                #     print(result['classification_report'])
                label_names_ = list(range(len(result['confusion_matrix']))) if self.label_names is None \
                    else self.label_names
                table = [[label_names_[i] + ' (true)'] + c for i, c in enumerate(result['confusion_matrix'])]
                print(tabulate(table, headers=label_names_))
                tn, fp, fn, tp = np.ravel(result['confusion_matrix'])
                print('Sensitivity = %.5f%%' % (100*tp / (tp + fn)))
                print('Specificity = %.5f%%' % (100*tn / (tn + fp)))

            # if verbose > 1:
            if hasattr(self.clf, 'votes'):
                print('votes:', *result['votes'] , sep='\n')
            if hasattr(self.clf, 'votes_pooled'):
                print('votes pooled:', result['votes_pooled'] )
            if self.is_grid:
                if isinstance(self.pipeline, MultiModalSeparateGridSearchCV):
                    for m in result['grid_scores'].keys():
                        for param, av, ste in result['grid_scores'][m]:
                            print('[%s] %s: %.5f +- %.5f' % (m, param, av, ste))
                    for m in result['grid_scores'].keys():
                        grid_params = [k for k in result.keys() if k.startswith('param__%s' % m)]
                        for grid_param in grid_params:
                            print('Best %s: %s' % (grid_param, result[grid_param]))
                else:
                    for param, av, ste in result['grid_scores']:
                        print('%s: %.5f +- %.5f' % (param, av, ste))
                    grid_params = [k for k in result.keys() if k.startswith('param__')]
                    for grid_param in grid_params:
                        print('Best %s: %s' % (grid_param, result[grid_param]))

            if 'weighting' in result:
                try:
                    print('\nWeighting:')
                    for m, w in result['weighting'].items():
                        print('\t%s: %.3f +- %.3f' % (m, np.mean(w), sem(w)))
                    weightings = [tuple([v[i] for v in result['weighting'].values()]) for i in range(self.n_folds)]
                    weightings_uniq = list(set(weightings))
                    counts = [weightings.count(x) for x in weightings_uniq]
                    # topn = min(5, len(weightings_uniq))
                    topn = len(weightings_uniq)
                    print('Top %g: %s\n' % (topn, str({weightings_uniq[i]: counts[i] for i in np.argsort(counts)[::-1][:topn]})[1:-1]))
                except Exception as e:
                    print('Catched error: %s' % str(e.args))
        if self.verbose > -1:
            if self.name:
                print('\tof << %s >>' % self.name)
            print('***************************************************')

    def _assess_performance(self, result, container=None):
        """ Computes performance measures for the model predictions

        Parameters
        ----------
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
        assert len(set([np.sum(np.isfinite(result[s]['y_pred'])) for s in range(n_seeds)])) == 1, \
            'Seeds appear to have different invalid predictions, which is not yet supported'
        valid = np.isfinite(result[0]['y_pred'])
        pred_ = np.mean([result[s]['y_pred'][valid] for s in range(n_seeds)],
                    axis=0)
        container['predictions'] = pred_.tolist()

        container['scoring_seed'] = \
            [self.scoring(self.y_true[valid], result[s]['y_pred'][valid] if self.is_regression
             else np.round(result[s]['y_pred'][valid])) for s in range(n_seeds)]
        container['scoring_ste'] = \
            np.std(container['scoring_seed'] / np.sqrt(n_seeds)).tolist()
        container['scoring'] = \
            self.scoring(self.y_true[valid], container['predictions'] if self.is_regression
            else np.round(container['predictions'])).tolist()

        if self.is_regression:
            container['correct'] = None
            container['explained_variance'] = \
                explained_variance_score(self.y_true[valid], container['predictions']).tolist()
        else:
            container['correct'] = (np.round(container['predictions']) == self.y_true[valid]).tolist()
            if not False in np.isfinite(container['predictions']):
                container['confusion_matrix'] = \
                    confusion_matrix(self.y_true[valid], np.round(container['predictions'])).tolist()
                container['p_balanced'] = bacc_p(np.array(container['confusion_matrix']))
                container['CI_balanced'] = bacc_ppi(np.array(container['confusion_matrix']))
                container['classification_report'] = \
                    classification_report(self.y_true[valid], np.round(container['predictions']),
                                          target_names=self.label_names,
                                          labels=np.unique(self.scheme.labels[valid]))

        return container

    def _forest_importances(self, X, test_cache, results, s):
        if isinstance(X, np.ndarray):
            features = np.ones(X.shape[1], dtype=bool)
            if 'feature_importance' not in results[s]:
                results[s]['feature_importance'] = np.full(X.shape[1], np.nan)
            features[features] = self.fs.get_support()
            contrib = ti.predict(self.clf, np.array(test_cache))[2]
            results[s]['feature_importance'][features] = \
                np.mean(contrib if self.is_regression else contrib[:, :, 0], axis=0)

        return results

    def _svc_importances(self, results, s):
        results[s]['feature_importance'] = np.squeeze(self.clf.coef_)
        return results

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

    def _construct_pipe(self):

        """ Construct the processing pipeline (masking, feature selection, classification)

        """

        self.is_multimodal = not (len(self.scheme.channels) == 1 and list(self.scheme.channels.keys())[0] == 'data')

        assert not (self.scheme.meta_clf is None and self.is_multimodal), \
            'If multiple channels are defined, a meta classifier needs to be defined'

        if hasattr(self.scheme, 'scoring') and self.scheme.scoring is not None:
            try:
                self.scoring = getattr(importlib.import_module('sklearn.metrics'), '%s_score' %
                                       self.scheme.scoring)
            except:
                self.scoring = getattr(importlib.import_module('decog.metrics'), '%s_score' %
                                       self.scheme.scoring)
            self.scoring_name = self.scheme.scoring
        else:
            self.scoring = r2_score if self.is_regression else accuracy_score
            self.scoring_name = 'r2' if self.is_regression else 'pred'

        self.param_grid, self.data = dict(), dict()
        if not self.is_multimodal:

            c = list(self.scheme.channels.keys())[0]
            channel = self.scheme.channels[c]
            self.data = channel.data.data
            self.channel = channel

            if channel.masker is not None:
                if 'verbose' in signature(channel.masker.masker).parameters.keys():
                    channel.masker.masker_args.update(verbose=self.verbose)
                if np.any([isinstance(v, list) for v in channel.masker.masker_args.values()]):
                    for k, v in channel.masker.masker_args.items():
                        if k != 'rois' and (isinstance(v, list) or isinstance(v, np.ndarray)):
                            if len(v) > 1:
                                self.param_grid['masker__%s' % k] = v
                            else:
                                channel.masker.masker_args[k] = v[0]

            if channel.fss is not None:
                if np.any([(isinstance(v, list) or (isinstance(v, np.ndarray)) and len(v) > 1)
                           for v in channel.fss.fs_args.values()]):
                    for k, v in channel.fss.fs_args.items():
                        if isinstance(v, list) or isinstance(v, np.ndarray):
                            if len(v) > 1:
                                self.param_grid['fs__%s' % k] = v
                            else:
                                channel.fss.fs_args[k] = v[0]

            if self.scheme.meta_clf is not None:
                if np.any([isinstance(v, list) or (isinstance(v, np.ndarray) and len(v) > 1)
                           for v in channel.clfs.clf_args.values()]):
                    for k, v in channel.clfs.clf_args.items():
                        if isinstance(v, list) or isinstance(v, np.ndarray):
                            if len(v) > 1:
                                self.param_grid['clf__base_estimator__%s' % k] = v
                            else:
                                channel.clfs.clf_args[k] = v[0]
            else:                
                if (self.n_jobs_folds is not None) and self.searchlight:
                    channel.clfs.clf_args.update(n_jobs=self.n_jobs_folds)
                mute_list = ["<class 'sklearn.svm.classes.%s'>" % v for v in ['SVC', 'SVR', 'LinearSVC']]
                if 'verbose' in signature(channel.clfs.clf).parameters.keys() and \
                                str(channel.clfs.clf) not in mute_list:
                    channel.clfs.clf_args.update(verbose=max(0, self.verbose-2))
                if np.any([((isinstance(v, list) or isinstance(v, np.ndarray) and len(v) > 1)) for v in channel.clfs.clf_args.values()]):
                    for k, v in channel.clfs.clf_args.items():
                        if isinstance(v, list) or isinstance(v, np.ndarray):
                            if len(v) > 1:
                                self.param_grid['clf__%s' % k] = v
                            else:
                                channel.clfs.clf_args[k] = v[0]

            pipes = []
            # define pipeline
            if channel.masker is not None:
                pipes.append(('masker', channel.masker.masker(**channel.masker.masker_args)))
            if channel.preproc is not None:
                preprocessor_args = dict() if channel.preproc.preprocessor_args is None\
                    else channel.preproc.preprocessor_args
                preproc = channel.preproc.preprocessor(**preprocessor_args)
                self.preproc = Pipeline([(channel.preproc.name, preproc)])
            if channel.fss is not None:
                fs = channel.fss.fs(**channel.fss.fs_args)
                pipes.append(('fs', fs))
            if self.scheme.meta_clf is None:
                self.clf = channel.clfs.clf(**channel.clfs.clf_args)
            else:
                be_dict = {'base_estimator': channel.clfs.clf(**channel.clfs.clf_args)}
                meta_clf_args = dict() if self.scheme.meta_clf_args is None else self.scheme.meta_clf_args
                meta_args = {**be_dict, **meta_clf_args}
                self.clf = self.scheme.meta_clf(**meta_args)

            pipes.append(('clf', self.clf))
            self.pipeline = Pipeline(pipes)
                          

        else:
            self.data = OrderedDict()
            self.channel = self.scheme.channels
            # define pipeline
            pipelines = dict()
            for c, channel in self.scheme.channels.items():
                self.data[c] = channel.data.data

                if np.any([isinstance(v, list) for v in channel.masker.masker_args.values()]):
                    for k, v in channel.masker.masker_args.items():
                        if k != 'rois' and (isinstance(v, list) or isinstance(v, np.ndarray)):
                            if len(v) > 1:
                                self.param_grid['union_%s__masker__%s' % (c, k)] = v
                            else:
                                channel.masker.masker_args[k] = v[0]
                if channel.fss is not None:
                    if np.any([isinstance(v, list) for v in channel.fss.fs_args.values()]):
                        for k, v in channel.fss.fs_args.items():
                            if isinstance(v, list) or isinstance(v, np.ndarray):
                                if len(v) > 1:
                                    self.param_grid['union_%s__fs__%s' % (c, k)] = v
                                else:
                                    channel.fss.fs_args[k] = v[0]

                pipes = [('masker', channel.masker.masker(**channel.masker.masker_args))]

                if channel.preproc is not None:
                    preprocessor_args = dict() if channel.preproc.preprocessor_args is None\
                        else channel.preproc.preprocessor_args
                    preproc = channel.preproc.preprocessor(**preprocessor_args)
                    self.preproc = Pipeline([(channel.preproc.name, preproc)])

                if channel.fss is not None:
                    fs = channel.fss.fs(**channel.fss.fs_args)
                    pipes.append(('fs', fs))
                pipelines[c] = Pipeline(pipes)

            for c, channel in self.scheme.channels.items():
                if np.any([((isinstance(v, list) or isinstance(v, np.ndarray) and len(v) > 1))
                           for v in channel.clfs.clf_args.values()]):
                    for k, v in channel.clfs.clf_args.items():
                        if isinstance(v, list) or isinstance(v, np.ndarray):
                            if len(v) > 1:
                                self.param_grid['clf__%s__%s' % (c, k)] = v
                            else:
                                channel.clfs.clf_args[k] = v[0]
            be_dict = {'base_estimators': {c: channel.clfs.clf(**channel.clfs.clf_args)
                                           for c, channel in self.scheme.channels.items()}}
            meta_clf_args = dict() if self.scheme.meta_clf_args is None else self.scheme.meta_clf_args
            # meta_clf_args.update(n_jobs=self.n_jobs_folds)
            meta_clf_args.update(verbose=max(0, self.verbose-1))
            self.clf = self.scheme.meta_clf(**{**be_dict, **meta_clf_args})

            self.pipeline = Pipeline([
                ('union', MultiModalFeatureUnion(list(pipelines.items()))),
                ('clf', self.clf)
            ])

        if len(self.param_grid):
            self.is_grid = True
            grid_kwargs = dict()
            if self.is_multimodal:
                # gs = MultiModalGridSearchCV
                if hasattr(self.scheme, 'grid_multimodal_cadence') and self.scheme.grid_multimodal_cadence:
                    gs = MultiModalGridSearchCV
                    grid_kwargs.update(cadence=self.scheme.grid_multimodal_cadence)
                else:
                    gs = MultiModalSeparateGridSearchCV
                    if hasattr(self.scheme, 'grid_multimodal_optimize_on_whole'):
                        grid_kwargs.update(optimize_on_whole=self.scheme.grid_multimodal_optimize_on_whole)
                    if hasattr(self.scheme, 'grid_multimodal_optimize_default'):
                        grid_kwargs.update(optimize_default=self.scheme.grid_multimodal_optimize_default)
                    if hasattr(self.scheme, 'grid_multimodal_optimize_default_loop'):
                        grid_kwargs.update(optimize_default_loop=self.scheme.grid_multimodal_optimize_default_loop)
            elif self.is_regression or (self.scheme.grid_scoring == 'balanced_accuracy' and isinstance(self.scheme.cv_grid, (LeaveOneOut, ExhaustiveLeave2Out))):
                gs = GridSearchCVLabels
            else:
                gs = GridSearchCV
            if self.scheme.grid_scoring is None:
                self.scheme.grid_scoring = 'r2' if self.is_regression else 'accuracy'
            self.pipeline = gs(self.pipeline, param_grid=self.param_grid, verbose=max(0, self.verbose-3), refit=True,
                               scoring=self.scheme.grid_scoring, cv=self.scheme.cv_grid, n_jobs=self.n_jobs_grid, **grid_kwargs)
        else:
            self.is_grid = False

        self.steps = self.pipeline.steps if hasattr(self.pipeline, 'steps') else self.pipeline.estimator.steps
        self.step_names = [s[0] for s in self.steps]
