import os
import time
from warnings import warn
from inspect import signature
from collections import OrderedDict, Sequence
from dataset import connect
import json
import multiprocessing_on_dill as multiprocessing
import numpy as np
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression, VarianceThreshold, SelectFromModel
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score, explained_variance_score, confusion_matrix, classification_report
from nilearn.input_data import NiftiMasker
from decereb.feature_selection import \
    MultiRoiVarianceThreshold, MuliRoiSelectPercentile, MultiRoiSelectFromModel, SelectRoisFromModel
from decereb.estimators import RoiEnsemble
from decereb.masker import DummyMasker, MultiRoiMasker
from mgutils.archiving import zip_directory_structure
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from treeinterpreter import treeinterpreter as ti
import warnings


class LinkDef:

    def __init__(self, analysis_def=None, info=None, db_key=None, detailed_save=False):

        if analysis_def is None:
            raise TypeError("keyword argument analysis_def is obligatory")
        self.analysis_def = analysis_def
        self.info = info
        if db_key is None:
            self.db_key = OrderedDict(type='unknown')
        else:
            self.db_key = db_key
        self.detailed_save = detailed_save


class AnalysisDef:

    def __init__(self, X=None, y=None, clf=None, clf_args=None, regression=False, fs=None, masker=None,
                 masker_args=None, seed_list=None, label_names=None):

        if X is None:
            raise TypeError("keyword argument X is obligatory")
        if y is None:
            raise TypeError("keyword argument y is obligatory")
        if clf is None:
            raise TypeError("keyword argument clf is obligatory")

        self.X = X
        self.y = y
        self.clf = clf
        self.clf_args = clf_args
        self.regression = regression
        self.fs = fs
        self.masker = masker
        self.masker_args = masker_args
        self.seed_list = seed_list
        if label_names is None:
            if not self.regression:
                self.label_names = ['class_%g' % cls for cls in np.unique(y)]
        else:
            self.label_names = label_names

    def set_fs(self, **params):
        self.fs = OrderedDict(params)


class Link:

    def __init__(self, linkdef=None, analysis=None, result=None):
        self.linkdef = linkdef
        self.analysis = analysis
        self.result = result

class Chain:

    def __init__(self, linkdef_list):
        self.linkdef_list = linkdef_list

    def run(self, test_mode=False, n_jobs=1, verbose=2, seed=None, output_path=None, recompute=False,
            skip_runerror=True, skip_ioerror=False, zip_code_dirs=None):
        """

        Args:
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
                              skip_ioerror, db_string, lockfile))
                chain.append(link)

        else:
            pool = multiprocessing.Pool(None if n_jobs == -1 else n_jobs)
            chain = pool.map(_link, [(n_jobs, verbose, seed, link_id, len(self.linkdef_list), linkdef, skip_runerror,
                                      skip_ioerror, db_string, lockfile)
                                     for link_id, linkdef in enumerate(self.linkdef_list)])
            pool.close()

        print('Failed links:')
        for i, link in enumerate(chain):
            if not link.analysis.success:
                print('%s' % self.linkdef_list[i].db_key)
                for message in link.analysis.messages:
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

    n_jobs, verbose, seed, link_id, chain_len, linkdef, skip_runerror, skip_ioerror, db_string, lockfile = params

    link_string = "Running chain link %g / %g %s\n%s" % (link_id + 1, chain_len, db_string, str(linkdef.db_key)[12:-1])
    link_string_short = "[%g/%g] %s" % (link_id + 1, chain_len, str(linkdef.db_key)[12:-1])

    an = Analysis(linkdef.analysis_def, name=link_string_short)

    an.t_start, an.t_stamp_start = time.strftime("%Y/%m/%d %H:%M:%S"), time.time()

    print("\n-------  %s  -------" % an.t_start)
    print(link_string)
    print("-------------------------------------\n")

    an.messages = []
    try:
        an.result = an.run(n_jobs=n_jobs, verbose=verbose, seed=seed)
    except Exception as ex:
        an.success = False
        an.result = False
        an.messages.append("An exception of type {0} occured. Arguments:\n{1!r}".format(type(ex).__name__, ex.args))
        if skip_runerror:
            warn(an.messages[-1] + '\n' + link_string_short)
        else:
            print('[RunErr] ' + link_string_short)
            raise  # re-raise
    finally:
        an.success = True
        an.messages.append("Link ran successfully")
        an.t_end, an.t_stamp_end = time.strftime("%Y/%m/%d %H:%M:%S"), time.time()
        an.t_dur = an.t_stamp_end - an.t_stamp_start
        result = {k: v for k, v in an.result.items() if linkdef.detailed_save or
                  (not k.endswith('_seed') and not k.startswith('forest_contrib'))}
        db_dict = OrderedDict([('identifier', str(linkdef.db_key)[12:-1]),
                               ('accuracy', result['accuracy']),
                               ('accuracy_ste', result['accuracy_ste']),
                               ('proc_time', an.t_dur),
                               ('result', json.dumps(result)),
                               ('info', json.dumps(linkdef.info) if linkdef.detailed_save else None),
                               ('t_start', an.t_start),
                               ('t_end', an.t_end),
                               ('timestamp_start', an.t_stamp_start),
                               ('timestamp_end', an.t_stamp_end),
                               ('success', an.success),
                               ('messages', ", ".join(["[" + m + "]" for m in an.messages]) if linkdef.detailed_save else None)
                               ] +
                              list(linkdef.db_key.items()))
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

    return Link(linkdef=linkdef, analysis=an, result=result)


class Analysis:

    def __init__(self, cfg, name=''):
        self.cfg = cfg
        self.name = name
        self.X = cfg.X
        self.y = cfg.y
        self.n_samples = len(cfg.X)

        self.masker = None
        self.selection = []
        self.clf = None
        self.pipeline = None
        self.steps = None
        self.param_grid = None

    def run(self, n_jobs=1, verbose=2, seed=None):

        self.construct_pipe(n_jobs=n_jobs, seed=seed, verbose=verbose)

        if self.cfg.regression:
            cv_outer = KFold(len(self.y), n_folds=min(int(len(self.y)/4.), 10), random_state=seed)
        else:
            cv_outer = StratifiedKFold(self.y, n_folds=min(int(len(self.y)/4.), 10), random_state=seed)
        # cv_outer = LeaveOneOut(len(self.y))

        seed_list = self.cfg.seed_list if self.cfg.seed_list is not None else [seed]
        n_seeds = len(seed_list)
        n_folds = len(cv_outer)

        y_pred = []
        y_true = []
        grid_scores_fold_seed=[[None] * n_seeds for _ in range(n_folds)]
        if isinstance(self.clf, (RandomForestRegressor, RandomForestClassifier)):
            n_features = self.X[0].shape[0]
            forest_contrib = np.full((n_folds, n_seeds, n_features), np.nan)
        elif isinstance(self.clf, RoiEnsemble) and isinstance(self.clf.base_estimator, (RandomForestRegressor, RandomForestClassifier)):
            forest_contrib = dict()
        result = dict(
            correct_seed=[[None] * n_seeds for _ in range(self.n_samples)],
            predictions_seed=[[None] * n_seeds for _ in range(self.n_samples)],
            labels_seed=[[None] * n_seeds for _ in range(self.n_samples)],
            votes_seed=[[None] * n_seeds for _ in range(self.n_samples)],
            votes_pooled_seed=[[None] * n_seeds for _ in range(self.n_samples)],
            grid_scores_seed=[None] * n_seeds,
            correct=[None] * self.n_samples,
            predictions=[None] * self.n_samples,
            labels=self.y,
            votes=[None] * self.n_samples,
            votes_pooled=[None] * self.n_samples,
            n_seeds=n_seeds
        )

        for f, (train_index, test_index) in enumerate(cv_outer):
            if verbose > 1:
                print('[%s] Fold %g / %g [%g seed iteration(s)]' % (time.strftime("%d.%m %H:%M:%S"), f + 1,
                                                                    len(cv_outer), n_seeds))
                if self.name:
                    print('\tof << %s >>' % self.name)
            images_train = [self.X[i] for i in train_index]
            labels_train = [self.y[i] for i in train_index]
            images_test = [self.X[i] for i in test_index]
            labels_test = [self.y[i] for i in test_index]

            # (1 / 2) If no GridSearchCV is used, we can prefit the preprocessing steps
            if self.param_grid is None:
                preproc_train = Pipeline(self.pipeline.steps[:-1]).fit_transform(images_train, labels_train)

            for j, seed_ in enumerate(seed_list):
                if verbose > 3:
                    print('seed %g / %g: %s' % (j + 1, n_seeds, seed_))
                if hasattr(self.clf, 'random_state'):
                    self.clf.random_state = seed_
                if 'fs_nested' in self.steps and hasattr(self.steps['fs_nested'].estimator, 'random_state'):
                    self.steps['fs_nested'].estimator.random_state = seed_
                if self.param_grid is None:
                    self.clf.fit(preproc_train, labels_train)
                else:  # (2 /2) else we have to fit the entire pipeline for each seed
                    self.pipeline.fit(images_train, labels_train)
                    # preproc_train = Pipeline(self.pipeline.steps[:-1]).fit_transform(images_train, labels_train)
                    # self.clf.fit(preproc_train, labels_train)
                # self.pipeline.fit(images_train, labels_train)
                preproc_test = Pipeline(self.pipeline.steps[:-1]).transform(images_test)
                predictions = self.clf.predict(preproc_test)
                # predictions = self.pipeline.predict(images_test)
                if isinstance(self.clf, (RandomForestRegressor, RandomForestClassifier)):
                    features = np.ones(n_features, dtype=bool)
                    for sel in self.selection:
                        features[features] = sel.get_support()
                    contrib = ti.predict(self.clf, np.array(preproc_test))[2]
                    forest_contrib[f, j, features] = np.mean(contrib if self.cfg.regression else contrib[:, :, 0], axis=0)
                elif isinstance(self.clf, RoiEnsemble) and isinstance(self.clf.base_estimator, (RandomForestRegressor, RandomForestClassifier)):
                    features = dict()
                    if self.selection:
                        for sel in self.selection:
                            for k, v in sel.get_support().items():
                                if k not in features:
                                    features[k] = v
                                else:
                                    features[k][features[k]] = v
                    else:
                        features = {k: np.ones(len(v[0]), dtype=bool) for k, v in preproc_test.items()}
                    for k in features.keys():
                        if k not in forest_contrib:
                            forest_contrib[k] = np.full((n_folds, n_seeds, len(features[k])), np.nan)
                        contrib = ti.predict(self.clf.estimators_[k], preproc_test[k])[2]
                        forest_contrib[k][f, j, features[k]] = np.mean(contrib if self.cfg.regression else contrib[:, :, 0], axis=0)
                y_pred = np.append(y_pred, predictions)
                y_true = np.append(y_true, labels_test)

                for i, k in enumerate(test_index):
                    result['predictions_seed'][k][j] = float(predictions[i])
                    result['labels_seed'][k][j] = labels_test[i]
                    result['correct_seed'][k][j] = bool(predictions[i] == labels_test[i])
                    if hasattr(self.clf, 'votes_pooled'):
                        result['votes_pooled_seed'][k][j] = self.clf.votes_pooled[i]
                    if hasattr(self.clf, 'votes'):
                        result['votes_seed'][k][j] = [v[i].tolist() for v in self.clf.votes]

                if self.param_grid is not None:
                    grid_scores_fold_seed[f][j] = [score.mean_validation_score for score in self.pipeline.grid_scores_]

            for k in test_index:
                result['predictions'][k] = np.mean(result['predictions_seed'][k], axis=0)
                result['correct'][k] = np.mean(result['correct_seed'][k], axis=0)
            if hasattr(self.clf, 'votes'):
                for k in test_index:
                    result['votes'][k] = np.mean(result['votes_seed'][k], axis=0).tolist()
            if hasattr(self.clf, 'votes_pooled'):
                for k in test_index:
                    result['votes_pooled'][k] = np.mean(result['votes_pooled_seed'][k], axis=0)

            if (verbose > 2) and hasattr(self.clf, 'votes'):
                print('votes:', *np.array([result['votes'][i] for i in test_index]).swapaxes(0, 1).tolist(), sep='\n')
            if (verbose > 2) and hasattr(self.clf, 'votes_pooled'):
                print('pooled votes: ', [result['votes_pooled'][i] for i in test_index])

        scorer = r2_score if self.cfg.regression else accuracy_score
        result['accuracy'] = scorer(y_true, y_pred)
        label_swap = np.swapaxes(result['labels_seed'], 0, 1)
        pred_swap = np.swapaxes(result['predictions_seed'], 0, 1)
        result['accuracy_seed'] = [scorer(l, p) for l, p in zip(label_swap, pred_swap)]
        result['accuracy_ste'] = np.std(result['accuracy_seed'] / np.sqrt(n_seeds))
        if isinstance(self.clf, (RandomForestRegressor, RandomForestClassifier)):
            with warnings.catch_warnings():  # catch stupid behavior of nanmean
                warnings.simplefilter("ignore", RuntimeWarning)
                result['forest_contrib_seed'] = np.nanmean(forest_contrib, axis=0).tolist()
                result['forest_contrib'] = np.nanmean(result['forest_contrib_seed'], axis=0).tolist()
        elif isinstance(self.clf, RoiEnsemble) and isinstance(self.clf.base_estimator, (RandomForestRegressor, RandomForestClassifier)):
            result['forest_contrib_seed'], result['forest_contrib'] = {}, {}
            with warnings.catch_warnings():  # catch stupid behavior of nanmean
                warnings.simplefilter("ignore", RuntimeWarning)
                for k in forest_contrib.keys():
                    result['forest_contrib_seed'][k] = np.nanmean(forest_contrib[k], axis=0).tolist()
                    result['forest_contrib'][k] = np.nanmean(result['forest_contrib_seed'][k], axis=0).tolist()

        if verbose:
            print('***************************************************')
            print("Accuracy: %.5f +- %.5f %s" % (result['accuracy'], result['accuracy_ste'],
                  ['%.5f' % acc for acc in result['accuracy_seed']]))
            if not self.cfg.regression:
                print(classification_report(y_true, y_pred, target_names=self.cfg.label_names))
            if self.name:
                print('\tof << %s >>' % self.name)
            print('***************************************************')

        if self.param_grid is not None:
            grid_scores_mean = np.mean(grid_scores_fold_seed, axis=(0, 1)).tolist()
            grid_scores_ste = (np.std(np.mean(grid_scores_fold_seed, axis=1), axis=0) / len(cv_outer)).tolist()
            result['grid_scores'] = [(param, grid_scores_mean[i], grid_scores_ste[i]) for i, param in
                                                  enumerate([score.parameters for score in self.pipeline.grid_scores_])]
            if verbose > 2:
                for param, av, ste in result['grid_scores']:
                    print('%s: %.5f +- %.5f' % (param, av, ste))

        return result

    def construct_pipe(self, n_jobs, seed, verbose):

        self.param_grid = dict()

        # Masker
        if not hasattr(self.cfg, 'masker_args') or (not hasattr(self.cfg, 'masker') and not self.cfg.masker):
            self.cfg.masker_args = dict()
        if hasattr(self.cfg, 'masker') and not self.cfg.masker:
            self.cfg.masker = DummyMasker

        if np.any([isinstance(v, list) for v in self.cfg.masker_args.values()]):
            for k, v in self.cfg.masker_args.items():
                if k != 'rois' and isinstance(v, list):
                    self.param_grid['masker__%s' % k] = v

        pipes = [('masker', self.cfg.masker(**self.cfg.masker_args))] if hasattr(self.cfg, 'masker') \
           else [('masker', NiftiMasker(**self.cfg['masker_args']))]
        self.masker = pipes[0][1]

        if 'smoothing_fwhm' in self.cfg.masker_args and isinstance(self.cfg.masker_args['smoothing_fwhm'], list):
            self.param_grid['masker__smoothing_fwhm'] = self.cfg.masker_args['smoothing_fwhm']

        fs_list = self.cfg.fs if isinstance(self.cfg.fs, Sequence) else (self.cfg.fs, )
        fs_types = [f.fs_type for f in fs_list]

        # variance-based feature selection
        if 'variance' in fs_types:
            fs = fs_list[fs_types.index('variance')]
            variance_threshold = MultiRoiVarianceThreshold() if isinstance(pipes[0][1], MultiRoiMasker) \
                            else VarianceThreshold()
            for k, v in fs.fs_args.items():
                if isinstance(v, list):
                    self.param_grid['fs_variance__%s' % k] = v
                else:
                    variance_threshold.set_params(**{k: v})
            pipes.append(('fs_variance', variance_threshold))
            self.selection.append(variance_threshold)

        # F-test-based feature selection
        if 'percentile' in fs_types:
            fs = fs_list[fs_types.index('percentile')]
            f_score = f_regression if self.cfg.regression else f_classif
            percentile = MuliRoiSelectPercentile(score_func=f_score) if \
                isinstance(pipes[0][1], MultiRoiMasker) else SelectPercentile(score_func=f_score)
            for k, v in fs.fs_args.items():
                if isinstance(v, list):
                    self.param_grid['fs_anova__%s' % k] = v
                else:
                    percentile.set_params(**{k: v})
            pipes.append(('fs_anova', percentile))
            self.selection.append(percentile)

        # model-based feature selection

        if 'model' in fs_types:
            fs = fs_list[fs_types.index('model')]
            if isinstance(pipes[0][1], MultiRoiMasker):
                select_from = MultiRoiSelectFromModel
            else:
                select_from = SelectFromModel
            if fs.fs_args['model'] == 'nested':
                nested_clf = self.cfg.clf(**self.cfg.clf_args)
                for k, v in self.cfg.clf_args.items():
                    if isinstance(v, list):
                        self.param_grid['fs_model__estimator__%s' % k] = v
                fs_model = select_from(nested_clf, threshold=fs.fs_args['threshold'])
            else:
                if 'model_args' not in fs.fs_args:
                    fs.fs_args['model_args'] = dict()
                for k, v in fs.fs_args['model_args'].items():
                    if isinstance(v, list):
                        self.param_grid['fs_model__estimator__%s' % k] = v
                fs_model = select_from(fs.fs_args['model'](**fs.fs_args['model_args']), threshold=fs.fs_args['threshold'])
            if isinstance(fs.fs_args['threshold'], list):
                self.param_grid['fs_model__threshold'] = fs.fs_args['threshold']

            pipes.append(('fs_model', fs_model))
            self.selection.append(fs_model)

        # model-based selection of ROIs
        if 'roi' in fs_types:
            fs = fs_list[fs_types.index('roi')]
            if fs.fs_args['roi_model'] == 'nested':
                for k, v in self.cfg.clf_args.items():
                    if isinstance(v, list):
                        self.param_grid['fs_roi__estimator__%s' % k] = v
                if 'base_estimator_args' in self.cfg.clf_args:
                    for k, v in self.cfg.clf_args['base_estimator_args'].items():
                        if isinstance(v, list):
                            self.param_grid['fs_roi__estimator__base_estimator__%s' % k] = v
                fs_roi = SelectRoisFromModel(self.cfg.clf(**self.cfg.clf_args), criterion=fs.fs_args['roi_criterion'])
            else:
                if 'roi_model_args' not in fs.fs_args:
                    fs.fs_args['roi_model_args'] = dict()
                for k, v in fs.fs_args['roi_model_args'].items():
                    if isinstance(v, list):
                        self.param_grid['fs_roi__estimator__%s' % k] = v
                fs_roi = SelectRoisFromModel(fs.fs_args['roi_model'](**fs.fs_args['roi_model_args']), criterion=fs.fs_args['roi_criterion'])
            if isinstance(fs.fs_args['roi_criterion'], list):
                self.param_grid['fs_roi__criterion'] = fs.fs_args['roi_criterion']
            pipes.append(('fs_roi', fs_roi))
            self.selection.append(fs_roi)


        # classifier
        if 'class_weight' in signature(self.cfg.clf.__init__).parameters.keys():
            self.cfg.clf_args.update(class_weight='balanced')
        self.clf = self.cfg.clf(**self.cfg.clf_args)
        if np.any([isinstance(v, list) for v in self.cfg.clf_args.values()]):
            for k, v in self.cfg.clf_args.items():
                if isinstance(v, list):
                    self.param_grid['clf__%s' % k] = v
        if 'base_estimator_args' in self.cfg.clf_args and \
                np.any([isinstance(v, list) for v in self.cfg.clf_args['base_estimator_args'].values()]):
            for k, v in self.cfg.clf_args['base_estimator_args'].items():
                if isinstance(v, list):
                    self.param_grid['clf__base_estimator__%s' % k] = v
        pipes.append(('clf', self.clf))

        if not len(self.param_grid):
            self.param_grid = None

        # create final pipeline and parameter grid
        self.pipeline = Pipeline(pipes)
        self.steps = OrderedDict(self.pipeline.steps)
        if self.param_grid is not None:
            self.pipeline = GridSearchCV(self.pipeline, param_grid=self.param_grid, n_jobs=n_jobs, verbose=verbose-3,
                                         refit=True, cv=3, scoring='accuracy')
