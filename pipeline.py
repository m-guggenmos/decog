import os
import time
from warnings import warn
from inspect import signature
from collections import OrderedDict
from dataset import connect
import json
import multiprocessing_on_dill as multiprocessing
import numpy as np
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression, VarianceThreshold, SelectFromModel
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score, explained_variance_score, confusion_matrix, classification_report
from decereb.feature_selection import \
    MultiRoiVarianceThreshold, MuliRoiSelectPercentile, MultiRoiSelectFromModel, SelectRoisFromModel
from mgutils.archiving import zip_directory_structure
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from treeinterpreter import treeinterpreter as ti
import warnings


class Link:

    def __init__(self, scheme=None):

        self.scheme = scheme

        self.info = dict()
        self.result = None
        self.analysis = None

        self.n_channels = len(self.scheme.pipeline)

        self.build_db_data()

    def build_db_data(self):

        self.db_key = OrderedDict(
            list(self.scheme.identifier.items()) +
            sum([[('%s%s' % (('', '%g_' % c)[self.n_channels > 1], k), v) for k, v in channel.clf.identifier.items()]
                 for c, channel in enumerate(self.scheme.pipeline)], []) +
            sum(sum([[[('%s%s' % (('', '%g_' % c)[self.n_channels > 1], k), v) for k, v in fs.identifier.items()] for fs in channel.fs] if channel.fs is not None else [[('%s%s' % (('', '%g_' % c)[self.n_channels > 1], 'fs_name'), None)]]
                     for c, channel in enumerate(self.scheme.pipeline)], []), [])
        )

        self.db_key_str = str(self.db_key)[12:-1]

        self.db_info = OrderedDict(
            sum([[('%s%s' % (('', '%g_' % c)[self.n_channels > 1], info), channel[info]) for c, channel in enumerate(self.scheme.data)] for info in ['labels', 'subjects', 'label_names', 'feature_names']], [])
            # sum([[('%g_%s' % (c, k), v) for k, v in channel.clf.opt_args.items()] for c, channel in enumerate(self.scheme.pipeline)], [])
        )


class Chain:

    def __init__(self, linkdef_list):
        self.linkdef_list = linkdef_list

    def run(self, test_mode=False, n_jobs=1, verbose=2, seed=None, output_path=None, recompute=False,
            skip_runerror=True, skip_ioerror=False, zip_code_dirs=None, detailed_save=False):
        """

        Args:
            zip_code_dirs:
            detailed_save:
            test_mode (bool): run in test-mode (enhanced debugging)
            n_jobs (int): number of jobs for multiprocessing
            verbose (int): verbosity level
            seed (None, int): seed for random number generator
            output_path: directoy in which analyses are saved as pickle files
            recompute (bool): recompute items present in the provided database (if one is provided)
            skip_runerror: skip classifier-related errors
            skip_ioerror: skip I/O errors
        """

        timestamp = time.strftime("%Y%m%d-%H%M%S")

        if output_path is None:
            db_string = 'sqlite:///:memory:'
            output_dir = '/tmp/'
        elif os.path.splitext(output_path)[1] == '.db':
            output_dir = os.path.splitext(output_path)[0]
            if os.path.exists(output_path) and not recompute:
                db_string = 'sqlite:///%s' % output_path
                db = connect(db_string)
                in_db = []
                for i, linkdef in enumerate(self.linkdef_list):
                    where = ' '.join(["AND %s IS '%s'" % (k, v) for k, v in linkdef.db_key.items()])[4:]
                    if list(db.query('SELECT id FROM chain WHERE %s' % where)):  # if entry exists (better way?)
                        in_db.append(i)
                for i in sorted(in_db, reverse=True):
                    print('deleting %s' % linkdef.db_key)
                    del self.linkdef_list[i] # if entry in database, remove from linkdef_list
            else:
                db_string = 'sqlite:///%s_%s.db' % (os.path.splitext(output_path)[0], timestamp)
        else:
            output_dir = output_path
            db_string = 'sqlite:///%s' % os.path.join(output_path, 'data_%s.db' % timestamp)

        if zip_code_dirs is not None:
            zip_directory_structure(zip_code_dirs, os.path.join(output_dir, 'archive_%s.zip' % timestamp),
                                    allowed_pattern='*.py')

        lockfile = os.path.join(output_dir, 'lock_%s' % timestamp)

        # chain = Parallel(n_jobs=n_jobs, backend="multiprocessing")(delayed(_link)(
        #     n_jobs, verbose, seed, link_id, len(self.linkdef_list), linkdef, skip_runerror, skip_ioerror, db_string, lockfile)
        #                                                      for link_id, linkdef in enumerate(self.linkdef_list))

        if test_mode:
            chain = []
            for link_id, linkdef in enumerate(self.linkdef_list):
                link = _link((n_jobs, verbose, seed, link_id, len(self.linkdef_list), linkdef, skip_runerror,
                              skip_ioerror, db_string, lockfile, detailed_save))
                chain.append(link)

        else:
            pool = multiprocessing.Pool(None if n_jobs == -1 else n_jobs)
            chain = pool.map(_link, [(n_jobs, verbose, seed, link_id, len(self.linkdef_list), linkdef, skip_runerror,
                                      skip_ioerror, db_string, lockfile, detailed_save)
                                     for link_id, linkdef in enumerate(self.linkdef_list)])
            pool.close()

        print('Failed links:')
        for i, link in enumerate(chain):
            if not link.info['success']:
                print('%s' % self.linkdef_list[i].db_key)
                for message in link.info['messages']:
                    print(message)

        return chain


def _link(params):
    """

    Args: `params` is a tuple consisting of (in this order):
        n_jobs (int): number of jobs for multiprocessing
        verbose (int): verbosity level
        seed (int): seed for random number generator
        link_id (int): integer id of the current link
        chain_len (int): length of the overall chain
        linkdef (dict): dict defining the current link
        skip_runerror: skip classifier-related errors
        skip_ioerror: skip I/O errors
        db_string (str): database connection string
        lockfile (str): path of lock file
    """

    n_jobs, verbose, seed, link_id, chain_len, link, skip_runerror, skip_ioerror, db_string, lockfile, detailed_save = params

    link_string = "Running chain link %g / %g %s\n%s" % (link_id + 1, chain_len, db_string, link.db_key_str)
    link_string_short = "[%g/%g] %s" % (link_id + 1, chain_len, link.db_key_str)

    link.analysis = Analysis(link.scheme, name=link_string_short)

    link.info['t_start'], link.info['t_stamp_start'] = time.strftime("%Y/%m/%d %H:%M:%S"), time.time()

    print("\n-------  %s  -------" % link.info['t_start'])
    print(link_string)
    print("-------------------------------------\n")

    link.info['messages'] = []
    try:
        link.result = link.analysis.run(n_jobs=n_jobs, verbose=verbose, seed=seed)
    except Exception as ex:
        link.info['success'] = False
        link.result = False
        link.info['messages'].append("An exception of type {0} occured. Arguments:\n{1!r}".format(type(ex).__name__, ex.args))
        if skip_runerror:
            warn(link.info['messages'][-1] + '\n' + link_string_short)
        else:
            print('[RunErr] ' + link_string_short)
            raise  # re-raise
    finally:
        link.info['success'] = True
        link.info['t_end'], link.info['t_stamp_end'] = time.strftime("%Y/%m/%d %H:%M:%S"), time.time()
        link.info['t_dur'] = link.info['t_stamp_end'] - link.info['t_stamp_start']
        result = {k: v for k, v in link.result.items() if detailed_save or not k.startswith('forest_contrib')}
        db_dict = OrderedDict([('accuracy', result['accuracy']),
                               ('accuracy_ste', result['accuracy_ste']),
                               ('proc_time', link.info['t_dur']),
                               ('result', json.dumps(result)),
                               ('info', json.dumps(link.db_info) if detailed_save else None),
                               ('t_start', link.info['t_start']),
                               ('t_end', link.info['t_end']),
                               ('timestamp_start', link.info['t_stamp_start']),
                               ('timestamp_end', link.info['t_stamp_end']),
                               ('success', link.info['success']),
                               ('messages', ", ".join(["[" + m + "]" for m in link.info['messages']]) if detailed_save else None)
                               ] +
                              list(link.db_key.items()))
    try:
        total_time = 0
        while os.path.exists(lockfile):
            time.sleep(0.1)
            total_time += 0.1
            if total_time > 100:
                raise IOError("Timeout reached for lock file\n" + link_string)
        open(lockfile, 'a').close()
        try:
            connect(db_string)['chain'].insert(db_dict)
        except Exception:
            if os.path.exists(lockfile):
                os.remove(lockfile)
            raise # re-raise
    except Exception as ex:
        if skip_ioerror:
            warn("An exception of type {0} occured. Arguments:\n{1!r}\n".format(type(ex).__name__, ex.args) +
                 link_string_short)
        else:
            print('[IOErr] ' + link_string_short)
            raise  # re-raise
    finally:
        if os.path.exists(lockfile):
            os.remove(lockfile)

    return link


class Analysis:

    def __init__(self, scheme, name=''):
        self.scheme = scheme
        self.name = name

        self.n_channels = len(self.scheme.data)
        self.multi_channel = self.n_channels > 1

        self.n_folds = None
        self.n_seeds = None
        self.n_samples = None
        self.is_regression = None
        self.y_true = None
        self.label_names = None
        self.scorer = None

        self.masker = None
        self.selection = None
        self.clf = None
        self.pipeline = None
        self.steps = None
        self.param_grid = None

    def run(self, n_jobs=1, verbose=2, seed=None):

        self.construct_pipe(n_jobs=n_jobs, seed=seed, verbose=verbose)

        for i, data in enumerate(self.scheme.data):
            assert np.array_equal(data.labels, self.scheme.data[0].labels), "Labels of all channels must be identical"
        for i, pipeline in enumerate(self.scheme.pipeline):
            assert pipeline.clf.regression == self.scheme.pipeline[0].clf.regression, \
                "All Channels must be either regressions or classifications"

        self.is_regression = self.scheme.pipeline[0].clf.regression
        self.y_true = np.array(self.scheme.data[0].labels)
        self.label_names = self.scheme.data[0].label_names
        self.scorer = r2_score if self.is_regression else accuracy_score

        seed_lists = [channel.clf.seed_list for channel in self.scheme.pipeline]
        self.n_seeds = [len(sl) for sl in seed_lists]
        self.n_samples = len(self.scheme.data[0].labels)

        if self.is_regression:
            cv_outer = KFold(self.n_samples, n_folds=min(int(self.n_samples/4.), 10), random_state=seed)
        else:
            cv_outer = StratifiedKFold(self.y_true, n_folds=min(int(self.n_samples/4.), 10), random_state=seed)
        # cv_outer = LeaveOneOut(len(self.y))

        self.n_folds = len(cv_outer)


        result_channels = [[None] * self.n_seeds[c] for c in range(self.n_channels)]
        for c in range(self.n_channels):
            for s in range(self.n_seeds[c]):
                result_channels[c][s] = dict(
                    y_pred=np.full(self.n_samples, np.nan),
                    grid_scores=[None] * self.n_folds
                )
                if self.n_channels > 1 and not self.is_regression:
                    result_channels[c][s].update(
                        y_pred_graded=np.full(self.n_samples, np.nan)
                    )
                if self.scheme.is_multiroi[c]:
                    result_channels[c][s].update(
                        votes=np.full((len(self.scheme.masker_args[0]['rois']), self.n_samples), np.nan),
                        votes_pooled=np.full(self.n_samples, np.nan),
                    )
        train_cache = None
        test_cache = None


        if train_cache is None:
            train_cache = [None] * self.n_folds
        if test_cache is None:
            test_cache = [None] * self.n_folds



        if self.n_channels > 1:
            result_meta = dict(
                y_pred=np.full(self.n_samples, np.nan)
            )

        for f, (train_index, test_index) in enumerate(cv_outer):

            if train_cache[f] is None:
                train_cache[f] = [None] * self.n_channels
            if test_cache[f] is None:
                test_cache[f] = [None] * self.n_channels

            for c in range(self.n_channels):

                X = self.scheme.data[c].data

                for s, seed_ in enumerate(seed_lists[c]):

                    if verbose > 1:
                        print('[%s] Fold %g / %g Channel %g / %g Seed %g / %g' %
                              (time.strftime("%d.%m %H:%M:%S"), f + 1, self.n_folds, c + 1, self.n_channels, s + 1, self.n_seeds[c]))
                        if self.name:
                            print('\tof << %s >>' % self.name)
                    images_train = [X[i] for i in train_index]
                    labels_train = [self.y_true[i] for i in train_index]
                    images_test = [X[i] for i in test_index]

                    # (1 / 2) If no GridSearchCV is used, we can prefit the preprocessing steps
                    if self.param_grid[c] is None and train_cache[f][c] is None:
                        train_cache[f][c] = Pipeline(self.steps[c][:-1]).fit_transform(images_train, labels_train)

                    if hasattr(self.clf[c], 'random_state'):
                        self.clf[c].random_state = seed_
                    if 'fs_nested' in self.steps and hasattr(self.steps[c]['fs_nested'].estimator, 'random_state'):
                        self.steps['fs_nested'].estimator.random_state = seed_
                    if self.param_grid[c] is None:
                        self.clf[c].fit(train_cache[f][c], labels_train)
                    else:  # (2 /2) else we have to fit the entire pipeline for each seed
                        self.pipeline[c].fit(images_train, labels_train)
                        # train_cache = Pipeline(self.pipeline.steps[:-1]).fit_transform(images_train, labels_train)
                        # self.clf.fit(train_cache, labels_train)

                        for step in range(len(self.steps[c])):
                            self.steps[c][step] = self.pipeline[c].best_estimator_.steps[step]
                        if len(self.selection[c]):
                            step_classes = [step[1].__class__ for step in self.pipeline[c].best_estimator_.steps]
                            for i, sel in enumerate(self.selection[c]):
                                self.selection[c][i] = self.pipeline[c].best_estimator_.steps[step_classes.index(sel.__class__)][1]
                        self.clf[c] = self.pipeline[c].best_estimator_._final_estimator
                        # self.pipeline[c] = self.pipeline[c].best_estimator_

                        train_cache[f][c] = Pipeline(self.steps[c][:-1]).transform(images_train)

                    if test_cache[f][c] is None:
                        test_cache[f][c] = Pipeline(self.steps[c][:-1]).transform(images_test)
                    predictions = self.clf[c].predict(test_cache[f][c])
                    if self.n_channels > 1 and not self.is_regression:
                        if hasattr(self.clf[c], 'predict_graded'):
                            predictions_graded = self.clf[c].predict_graded(test_cache[f][c])
                        else:
                            predictions_graded = predictions
                        # if hasattr(self.clf[c], 'predict_proba'):
                        #     predictions_graded_ = self.clf[c].predict_proba(test_cache[f][c])
                        #     predictions_graded = predictions_graded_[:, 1] - 0.5
                        # else:
                        #     predictions_graded = predictions - np.mean(np.unique(self.scheme.data[c].labels))
                    if self.param_grid[c] is not None:
                        result_channels[c][s]['grid_scores'][f] = \
                            [score.mean_validation_score for score in self.pipeline[c].grid_scores_]
                    if isinstance(self.clf[c], (RandomForestRegressor, RandomForestClassifier)) and isinstance(X, np.ndarray):
                        features = np.ones(X.shape[1], dtype=bool)
                        if 'feature_importance' not in result_channels[c][s]:
                            result_channels[c][s]['feature_importance'] = np.full((self.n_folds, X.shape[1]), np.nan)
                        for sel in self.selection[c]:
                            features[features] = sel.get_support()
                        contrib = ti.predict(self.clf[c], np.array(test_cache[f][c]))[2]
                        result_channels[c][s]['feature_importance'][f, features] = \
                            np.mean(contrib if self.is_regression else contrib[:, :, 0], axis=0)
                    elif self.scheme.is_multiroi[c] and isinstance(self.clf[c].base_estimator, (RandomForestRegressor, RandomForestClassifier)):
                        if 'feature_importance' not in result_channels[c][s]:
                            result_channels[c][s]['feature_importance'] = dict()
                        if self.selection:
                            features = dict()
                            for sel in self.selection[c]:
                                for k, v in sel.get_support().items():
                                    if k in test_cache[f][c].keys():
                                        if k not in features:
                                            features[k] = v
                                        else:
                                            features[k][features[k]] = v
                        else:
                            features = {k: np.ones(len(v[0]), dtype=bool) for k, v in test_cache[f][c].items()}
                        for k in test_cache[f][c].keys():
                            if k not in result_channels[c][s]['feature_importance']:
                                result_channels[c][s]['feature_importance'][k] = np.full((self.n_folds, len(features[k])), np.nan)
                            contrib = ti.predict(self.clf[c].estimators_[k], test_cache[f][c][k])[2]
                            result_channels[c][s]['feature_importance'][k][f, features[k]] = \
                                np.mean(contrib if self.is_regression else contrib[:, :, 0], axis=0)

                    result_channels[c][s]['y_pred'][test_index] = predictions.astype(float)
                    if self.n_channels > 1 and not self.is_regression:
                        result_channels[c][s]['y_pred_graded'][test_index] = predictions_graded.astype(float)
                    if hasattr(self.clf[c], 'votes_pooled'):
                        result_channels[c][s]['votes_pooled'][test_index] = self.clf[c].votes_pooled
                    if hasattr(self.clf[c], 'votes'):
                        for i, k in enumerate(self.clf[c].estimators_.keys()):
                            result_channels[c][s]['votes'][k, test_index] = np.array(self.clf[c].votes)[i, :]

            if self.n_channels > 1:
                result_meta['y_pred'][test_index] = self.meta_predict(
                    [np.mean([s[('y_pred_graded', 'y_pred')[self.is_regression]][test_index] for s in ch], axis=0) for i, ch in enumerate(result_channels)])
                # result_meta[s]['y_pred'][test_index] = \
                #     self.meta_predict(self.y_true[train_index],
                #                       [self.clf[ch].predict(train_cache[f][ch]) for ch in range(self.n_channels)],
                #                       [channel[s]['y_pred'][test_index] for channel in result_channels])

        result = dict()
        for c in range(self.n_channels):
            pfx = '%g_' % c if self.multi_channel else ''
            result = self.assess_performance(pfx, result_channels[c], container=result)

            if hasattr(self.clf[c], 'votes'):
                result[pfx + 'votes'] = np.nanmean([result_channels[c][s]['votes'] for s in range(self.n_seeds[c])], axis=0).tolist()
            if hasattr(self.clf[c], 'votes_pooled'):
                result[pfx + 'votes_pooled'] = np.mean([result_channels[c][s]['votes_pooled'] for s in range(self.n_seeds[c])], axis=0).tolist()

            if isinstance(self.clf[c], (RandomForestRegressor, RandomForestClassifier)) and isinstance(X, np.ndarray):
                with warnings.catch_warnings():  # catch stupid behavior of nanmean
                    warnings.simplefilter("ignore", RuntimeWarning)
                    result[pfx + 'forest_contrib'] = np.nanmean([result_channels[c][s]['feature_importance'] for s in range(self.n_seeds[c])], axis=(0, 1)).tolist()
            elif self.scheme.is_multiroi[c] and isinstance(self.clf[c].base_estimator, (RandomForestRegressor, RandomForestClassifier)):
                result[pfx + 'forest_contrib'] = {}
                with warnings.catch_warnings():  # catch stupid behavior of nanmean
                    warnings.simplefilter("ignore", RuntimeWarning)
                    for k in result_channels[0][c]['feature_importance'].keys():
                        result[pfx + 'forest_contrib'][k] = np.mean([result_channels[c][s]['feature_importance'][k] for s in range(self.n_seeds[c])], axis=(0,1)).tolist()

            if self.param_grid[c] is not None:
                grid_scores_mean = np.mean([result_channels[c][s]['grid_scores'] for s in range(self.n_seeds[c])], axis=(0, 1)).tolist()
                grid_scores_ste = (np.std(np.mean([result_channels[c][s]['grid_scores'] for s in range(self.n_seeds[c])], axis=1), axis=0) / self.n_folds).tolist()
                result[pfx + 'grid_scores'] = [(param, grid_scores_mean[i], grid_scores_ste[i]) for i, param in
                                                      enumerate([score.parameters for score in self.pipeline[c].grid_scores_])]

        if self.n_channels > 1:
            result = self.assess_performance('', [result_meta], container=result)
            print('meta:', result['predictions'])

        if verbose:
            print('***************************************************')
            for c in range(self.n_channels):
                pfx = '%g_' % c if self.multi_channel else ''

                print('***********       Channel (%g / %g)       ***********' % (c + 1, self.n_channels))
                print("Accuracy: %.5f +- %.5f %s" % (result[pfx + 'accuracy'], result[pfx + 'accuracy_ste'],
                      ['%.5f' % acc for acc in result[pfx + 'accuracy_seed']]))
                if self.is_regression:
                    print('Explained variance: %.4f%%' % (100 * result[pfx + 'explained_variance']))
                else:
                    print(result[pfx + 'classification_report'])

                if verbose > 2:
                    if hasattr(self.clf[c], 'votes'):
                        print('votes:', *np.mean([result_channels[c][s]['votes'] for s in range(self.n_seeds[c])], axis=0), sep='\n')
                    if hasattr(self.clf[c], 'votes_pooled'):
                        print('votes pooled:', np.mean([result_channels[c][s]['votes_pooled'] for s in range(self.n_seeds[c])], axis=0))
                    if self.param_grid[c] is not None:
                        for param, av, ste in result[pfx + 'grid_scores']:
                            print('%s: %.5f +- %.5f' % (param, av, ste))
            if self.n_channels > 1:
                print('***********       Combined Results       ***********')
                print("Accuracy: %.5f +- %.5f %s" % (result['accuracy'], result['accuracy_ste'],
                      ['%.5f' % acc for acc in result['accuracy_seed']]))
                if self.is_regression:
                    print('Explained variance: %.4f%%' % 100 * result['explained_variance'])
                else:
                    print(result['classification_report'])
            if self.name:
                print('\tof << %s >>' % self.name)
            print('***************************************************')

        return result

    def assess_performance(self, pfx, result, container=None):
        n_seeds = len(result)
        if container is None:
            container = dict()
        container[pfx + 'predictions'] = np.mean([result[s]['y_pred'] for s in range(n_seeds)], axis=0).tolist()

        container[pfx + 'accuracy_seed'] = [self.scorer(self.y_true, result[s]['y_pred'] if self.is_regression else np.round(result[s]['y_pred'])) for s in range(n_seeds)]
        container[pfx + 'accuracy_ste'] = np.std(container[pfx + 'accuracy_seed'] / np.sqrt(n_seeds)).tolist()
        container[pfx + 'accuracy'] = np.mean(self.y_true == np.round(container[pfx + 'predictions'])).tolist()

        if self.is_regression:
            container[pfx + 'explained_variance'] = explained_variance_score(self.y_true, container[pfx + 'predictions']).tolist()
        else:
            container[pfx + 'confusion_matrix'] = confusion_matrix(self.y_true, np.round(container[pfx + 'predictions'])).tolist()
            container[pfx + 'classification_report'] = \
                classification_report(self.y_true, np.round(container[pfx + 'predictions']),
                                      target_names=self.label_names)

        return container

    def meta_predict(self, y_pred_test_list):

        if self.scheme.clf_meta_args['weighting'] is not None:
            weighting = self.scheme.clf_meta_args['weighting']
            weighted_mean = np.array(y_pred_test_list).swapaxes(0, 1).dot(weighting) / np.sum(weighting)
        else:
            weighted_mean = np.mean(y_pred_test_list, axis=0)

        if self.is_regression:
            return weighted_mean
        else:
            # return np.unique(self.scheme.data[0].labels)[((1 + np.sign(weighted_mean)) / 2).astype(int)]
            return np.round(weighted_mean)

    def construct_pipe(self, n_jobs, seed, verbose):

        self.masker, self.clf, self.param_grid, self.steps, self.pipeline = [None] * self.n_channels, \
                  [None] * self.n_channels, [None] * self.n_channels, [None] * self.n_channels, [None] * self.n_channels
        self.selection = [[] for _ in range(self.n_channels)]

        for c in range(self.n_channels):

            self.param_grid[c] = dict()

            if np.any([isinstance(v, list) for v in self.scheme.masker_args[c].values()]):
                for k, v in self.scheme.masker_args[c].items():
                    if k != 'rois' and isinstance(v, list):
                        self.param_grid[c]['masker__%s' % k] = v

            pipes = [('masker', self.scheme.masker[c](**self.scheme.masker_args[c]))]
            self.masker[c] = pipes[0][1]

            if 'smoothing_fwhm' in self.scheme.masker_args[c] and isinstance(self.scheme.masker_args[c]['smoothing_fwhm'], list):
                self.param_grid[c]['masker__smoothing_fwhm'] = self.scheme.masker_args[c]['smoothing_fwhm']

            fs_names = [f.name for f in self.scheme.pipeline[c].fs] if self.scheme.pipeline[c].fs is not None else []

            # variance-based feature selection
            if 'variance' in fs_names:
                fs = self.scheme.pipeline[c].fs[fs_names.index('variance')]
                variance_threshold = MultiRoiVarianceThreshold() if self.scheme.is_multiroi[c] \
                                else VarianceThreshold()
                for k, v in fs.fs_args.items():
                    if isinstance(v, list):
                        self.param_grid[c]['fs_variance__%s' % k] = v
                    else:
                        variance_threshold.set_params(**{k: v})
                pipes.append(('fs_variance', variance_threshold))
                self.selection[c].append(variance_threshold)

            # F-test-based feature selection
            if 'percentile' in fs_names:
                fs = self.scheme.pipeline[c].fs[fs_names.index('percentile')]
                f_score = f_regression if self.scheme.pipeline[c].clf.regression else f_classif
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
            if 'model' in fs_names or 'nested' in fs_names:
                fs = self.scheme.pipeline[c].fs[fs_names.index(('nested', 'model')['model' in fs_names])]
                if self.scheme.is_multiroi[c]:
                    select_from = MultiRoiSelectFromModel
                else:
                    select_from = SelectFromModel
                if fs.fs_args['model'] == 'nested':
                    nested_clf = self.scheme.pipeline[c].clf.clf(**self.scheme.pipeline[c].clf.clf_args)
                    for k, v in self.scheme.pipeline[c].clf.clf_args.items():
                        if isinstance(v, list):
                            self.param_grid[c]['fs_model__estimator__%s' % k] = v
                    fs_model = select_from(nested_clf, threshold=fs.fs_args['threshold'])
                else:
                    if 'model_args' not in fs.fs_args:
                        fs.fs_args['model_args'] = dict()
                    for k, v in fs.fs_args['model_args'].items():
                        if isinstance(v, list):
                            self.param_grid[c]['fs_model__estimator__%s' % k] = v
                    fs_model = select_from(fs.fs_args['model'](**fs.fs_args['model_args']), threshold=fs.fs_args['threshold'])
                if isinstance(fs.fs_args['threshold'], list):
                    self.param_grid[c]['fs_model__threshold'] = fs.fs_args['threshold']

                pipes.append(('fs_model', fs_model))
                self.selection[c].append(fs_model)

            # model-based selection of ROIs
            if 'roi' in fs_names:
                fs = self.scheme.pipeline[c].fs[fs_names.index('roi')]
                if fs.fs_args['roi_model'] == 'nested':
                    for k, v in self.scheme.pipeline[c].clf.clf_args.items():
                        if isinstance(v, list):
                            self.param_grid[c]['fs_roi__estimator__%s' % k] = v
                    if 'base_estimator_args' in self.scheme.pipeline[c].clf.clf_args:
                        for k, v in self.scheme.pipeline[c].clf.clf_args['base_estimator_args'].items():
                            if isinstance(v, list):
                                self.param_grid[c]['fs_roi__estimator__base_estimator__%s' % k] = v
                    fs_roi = SelectRoisFromModel(self.scheme.pipeline[c].clf.clf(**self.scheme.pipeline[c].clf.clf_args),
                                                 criterion=fs.fs_args['roi_criterion'])
                else:
                    if 'roi_model_args' not in fs.fs_args:
                        fs.fs_args['roi_model_args'] = dict()
                    for k, v in fs.fs_args['roi_model_args'].items():
                        if isinstance(v, list):
                            self.param_grid[c]['fs_roi__estimator__%s' % k] = v
                    fs_roi = SelectRoisFromModel(fs.fs_args['roi_model'](**fs.fs_args['roi_model_args']), criterion=fs.fs_args['roi_criterion'])
                if isinstance(fs.fs_args['roi_criterion'], list):
                    self.param_grid[c]['fs_roi__criterion'] = fs.fs_args['roi_criterion']
                pipes.append(('fs_roi', fs_roi))
                self.selection[c].append(fs_roi)


            # classifier
            if 'class_weight' in signature(self.scheme.pipeline[c].clf.clf.__init__).parameters.keys():
                self.scheme.pipeline[c].clf.clf_args.update(class_weight='balanced')
            if self.scheme.is_multiroi[c]:
                be_dict = {'base_estimator': self.scheme.pipeline[c].clf.clf(**self.scheme.pipeline[c].clf.clf_args)}
                self.clf[c] = self.scheme.clf_multiroi[c](**{**be_dict, **self.scheme.clf_multiroi_args[c]})
                if np.any([isinstance(v, list) for v in self.scheme.pipeline[c].clf.clf_args.values()]):
                    for k, v in self.scheme.pipeline[c].clf.clf_args.items():
                        if isinstance(v, list):
                            self.param_grid[c]['clf__base_estimator__%s' % k] = v
                # if 'base_estimator_args' in self.scheme.pipeline[c].clf.clf_args and \
                #         np.any([isinstance(v, list) for v in self.scheme.pipeline[c].clf.clf_args['base_estimator_args'].values()]):
                #     for k, v in self.scheme.pipeline[c].clf.clf_args['base_estimator_args'].items():
                #         if isinstance(v, list):
                #             self.param_grid[c]['clf__base_estimator__%s' % k] = v
            else:
                self.clf[c] = self.scheme.pipeline[c].clf.clf(**self.scheme.pipeline[c].clf.clf_args)
                if np.any([isinstance(v, list) for v in self.scheme.pipeline[c].clf.clf_args.values()]):
                    for k, v in self.scheme.pipeline[c].clf.clf_args.items():
                        if isinstance(v, list):
                            self.param_grid[c]['clf__%s' % k] = v
            pipes.append(('clf', self.clf[c]))

            if not len(self.param_grid[c]):
                self.param_grid[c] = None

            # create final pipeline and parameter grid
            self.pipeline[c] = Pipeline(pipes)
            self.steps[c] = self.pipeline[c].steps
            if self.param_grid[c] is not None:
                self.pipeline[c] = GridSearchCV(self.pipeline[c], param_grid=self.param_grid[c], n_jobs=n_jobs, verbose=verbose-3,
                                             refit=True, cv=3, scoring='accuracy')
