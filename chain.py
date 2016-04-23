from collections import OrderedDict
import time
import pickle
from dataset import connect
import os
import multiprocessing_on_dill as multiprocessing

import json
from warnings import warn
import nibabel
from sklearn.svm import SVC
import numpy as np

from .util.archiving import zip_directory_structure
from .analysis import Analysis
from .descriptor import DescriptorConcatenator, Data, ClassifierDescriptor, \
    FeatureSelectionDescriptor, SchemeDescriptor, Channel


class SimpleChain:

    def __init__(self, data, clf=SVC, clf_args=None, fs=None, fs_args=None, labels=None, cv=None):

        """ Simple interface to start analyses.

        Parameters
        ----------
        data : List<NiftiImages>, np.ndarray, decereb.descriptor.data
        clf : sklearn type classifier class, decereb.descriptor.ClassifierDescriptor
        clf_args : None, dict
            Arguments passed to the classifier
        fs : None, str, decereb.descriptor.FeatureSelectionDescriptor
        fs_args : None, dict
            Arguments passed to the feature selection
        labels : List<int, str>
            only necessary if type(data) != decereb.descriptor.data
        """

        self.labels = labels
        self.data = data
        self.clf_args = clf_args
        self.clf = clf
        self.fs_args = fs_args
        self.fs = fs
        self.cv = cv

        self._build_chain()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if isinstance(value, list) and isinstance(value[0], Data):
            self._data = value
        elif isinstance(value, Data):
            self._data = [value]
        else:
            if self.labels is None:
                raise ValueError('No labels have been provided!')
            self._data = [Data(value, self.labels)]
        if hasattr(self, 'chain'):
            self._build_chain()

    @property
    def clf(self):
        return self._clf

    @clf.setter
    def clf(self, value):
        if isinstance(value, ClassifierDescriptor):
            self._clf = value
        else:
            self._clf = ClassifierDescriptor(name=str(value)[8:-2], clf=value, clf_args=self.clf_args)
        if hasattr(self, 'chain'):
            self._build_chain()

    @property
    def clf_args(self):
        return self._clf_args

    @clf_args.setter
    def clf_args(self, value):
        if value is None:
            self._clf_args = dict()
        else:
            self._clf_args = value
        if hasattr(self, 'chain'):
            self._build_chain()

    @property
    def fs(self):
        return self._fs

    @fs.setter
    def fs(self, value):
        if isinstance(value, FeatureSelectionDescriptor):
            self._fs = value
        else:
            self._fs = FeatureSelectionDescriptor(type=value, fs_args=self.fs_args)
        if hasattr(self, 'chain'):
            self._build_chain()

    def run(self, n_jobs_links=1, n_jobs_folds=1, verbose=1,
            output_path='/tmp/decereb/', skip_ioerror=False, skip_runerror=False,
            detailed_save=True):
        return self.chain.run(n_jobs_links, n_jobs_folds, verbose, output_path, skip_ioerror,
                              skip_runerror, detailed_save)

    def _build_chain(self):
        self.channel = Channel(clfs=self.clf, fss=self.fs)
        self.schemes = []
        for i, data in enumerate(self.data):
            cv = self.cv[i] if isinstance(self.cv, list) else self.cv
            self.schemes.append(SchemeDescriptor(name='%s_%g' % (data.name, i), data=data,
                                                 channels=self.channel, cv=cv))
        self.chain = ChainBuilder(schemes=self.schemes).build_chain()

class Link:

    def __init__(self, scheme=None):

        """ A link contains all the metainfo necessary for a single analysis. Links are
        concatenated by means of a decereb.chain.ChainBuilder to form a decereb.chain.Chain of
        analyses.

        Parameters
        ----------
        scheme : decereb.descriptor.SchemeDescriptor
            each link has to be initialized with a SchemeDescriptor

        """
        self.scheme = scheme

        self.info = dict()
        self.result = None
        self.analysis = None

        self.description = None
        self.description_str = None
        self.db_scheme_info_optional = None

        self.n_channels = len(self.scheme.channels)

        self.build_db_data()

    def build_db_data(self):

        """ Generate the database information to be saved for the current link
        """
        is_multichannel = self.n_channels > 1

        def prefix(c, is_multichannel_): return ('', '%g_' % c)[is_multichannel_]

        scheme_descrip = list(self.scheme.identifier.items())
        clf_descrip = [[('%s%s' % (prefix(c, is_multichannel), k), v)
                        for k, v in channel.clfs.identifier.items()]
                       for c, channel in enumerate(self.scheme.channels)]
        fs_descrip = [[[('%s%s' % (prefix(c, is_multichannel), k), v)
                        for k, v in fs.identifier.items()] for fs in channel.fss]
                      if channel.fss is not None
                      else [[('%s%s' % (prefix(c, is_multichannel), 'fs_name'), None)]]
                      for c, channel in enumerate(self.scheme.channels)]
        self.description = OrderedDict(
                scheme_descrip +
                sum(clf_descrip, []) +
                sum(sum(fs_descrip, []), [])
        )
        self.description_str = str(self.description)[12:-1]
        data_descrip = [[('%s%s' % (prefix(c, is_multichannel), info), channel[info])
                         for c, channel in enumerate(self.scheme.data)]
                        for info in Data.optional_info_items]
        self.description_optional = OrderedDict(sum(data_descrip, []))


class Chain:

    def __init__(self, link_list):
        """ The chain represents the entirety of analyses and distributes different sub-analyses
         (i.e. decereb.chain.Links) to different processors, if n_jobs_links > 1.

        Parameters
        ----------
        link_list : a list of decereb.chain.Link objects
        """
        self.link_list = link_list

    def run(self, n_jobs_links=1, n_jobs_folds=1, verbose=2, output_path=None, recompute=False,
            skip_runerror=True, skip_ioerror=False, zip_code_dirs=None, detailed_save=False):
        """ Start the analyses chain
        Parameters
        ----------
        n_jobs_links:  int
            number of parallell jobs for links
        n_jobs_folds: int
            number of parallell jobs for folds
        verbose: int
            verbosity level
        output_path : str, None
            directoy in which results are saved as pickle files OR
            path to existent database OR
            None, in which case results are stored in memory
        recompute : bool
            recompute links that are already in the provided database (if one is provided via
            output_path)
        skip_runerror : bool
            skip any runtime errors
        skip_ioerror:
            skip I/O errors related to saving the results in a database
        zip_code_dirs: List<str>, None
            python files encontained in a list of directory paths are saved as a zip file
        detailed_save : bool
            whether to save harddisk-intensive analyses results are saved
            [TODO: which ones?]
        """

        timestamp = time.strftime("%Y%m%d-%H%M%S")

        if n_jobs_links > 1 and n_jobs_folds > 1:
            raise Exception('You cannot set both n_jobs_chain and n_jobs_folds > 1.')


        is_searchlight = self.link_list[0].scheme.channels[0].clfs._searchlight
        has_time = self.link_list[0].scheme.data[0].has_time

        lockfile_timestamp = timestamp
        if not is_searchlight and not has_time:
            if output_path is None:
                db_string = 'sqlite:///:memory:'
                output_dir = '/tmp/'
            elif os.path.splitext(output_path)[1] == '.db':
                output_dir = os.path.dirname(output_path)
                if os.path.exists(output_path):
                    db_string = 'sqlite:///%s' % output_path
                    db = connect(db_string)
                    basename = os.path.splitext(os.path.basename(output_path))[0]
                    if len(basename) == 20 and basename.startswith('data_'):
                        lockfile_timestamp = basename[5:]
                    else:
                        print('Cannot infer lockfile name of previous chain(s). Using '
                              'timestamp as name instead.')
                    if not recompute:
                        in_db = []
                        exclude_keys = ['in_id', 'clf_id']
                        for i, link in enumerate(self.link_list):
                            if all([k in db[db.tables[0]].columns for k in link.description.keys()]):
                                where = ' '.join(["AND %s IS '%s'" % (k, v)
                                                  for k, v in link.description.items()
                                                  if k not in exclude_keys])[4:]
                                where = where.replace("'True'", "1").replace("'False'", "0").\
                                    replace("'None'", "NULL")
                                if list(db.query('SELECT id FROM chain WHERE %s' % where)):
                                    # and not list(db.query('SELECT correct FROM chain WHERE %s' % where))[0]['correct'] is None:
                                    in_db.append(i)
                        for i in sorted(in_db, reverse=True):
                            print('Deleting %s' % self.link_list[i].description_str)
                            del self.link_list[i]  # if entry in database, remove from linkdef_list
                        print('Deleted %g links' % len(in_db))
                else:
                    # db_string = 'sqlite:///%s_%s.db' % (os.path.splitext(output_dir)[0], timestamp)
                    db_string = 'sqlite:///%s' % output_path
            else:
                output_dir = output_path
                db_string = 'sqlite:///%s' % os.path.join(output_path, 'data_%s.db' % timestamp)
        else:
            db_string = ''
            if output_path is not None:
                output_dir = os.path.dirname(output_path)
            else:
                output_dir = None


        if output_dir is not None and zip_code_dirs is not None:
            zip_path = os.path.join(output_dir, 'archive_%s.zip' % timestamp)
            zip_directory_structure(zip_code_dirs, zip_path, allowed_pattern='*.py')
        if output_dir is not None and not has_time and not is_searchlight:
            lockfile = os.path.join(output_dir, 'lock_%s' % lockfile_timestamp)
            print('Lockfile: %s' % lockfile)
        else:
            lockfile = None
        if db_string:
            print('Database: %s' % db_string)

        if n_jobs_links == 1:
            chain = []
            for link_id, link in enumerate(self.link_list):
                params = (n_jobs_folds, verbose, link_id, len(self.link_list),
                          link, skip_runerror, skip_ioerror, db_string, lockfile, output_path,
                          timestamp, detailed_save)
                link = _link(params)
                chain.append(link)
        else:
            pool = multiprocessing.Pool(None if n_jobs_links == -1 else n_jobs_links)
            params = [(n_jobs_folds, verbose, link_id, len(self.link_list),
                       link, skip_runerror, skip_ioerror, db_string, lockfile, output_path,
                       timestamp, detailed_save)
                      for link_id, link in enumerate(self.link_list)]
            chain = pool.map(_link, params)
            pool.close()

        for i, link in enumerate(chain):
            if not link.info['success']:
                print('Failed link: %s' % self.link_list[i].db_key)
                for message in link.info['messages']:
                    print(message)

        print('Finished chain!')

        return chain[0] if len(chain) == 1 else chain


def _link(params):
    """ run link analysis and save results in the database

    Parameters
    ----------
    params : tuple
        tuple of parameters as passed by decereb.Chain.run()

    Returns
    -------
    link : decereb.chain.Link
        processed Link

    """
    n_jobs_folds, verbose, link_id, chain_len, link, skip_runerror, \
        skip_ioerror, db_string, lockfile, output_path, timestamp, detailed_save = params

    searchlight = link.scheme.channels[0].clfs._searchlight
    has_time = link.scheme.data[0].has_time

    link_string = "Running chain link %g / %g %s\n%s" % \
                  (link_id + 1, chain_len, db_string, link.description_str)
    link_string_short = "[%g/%g] %s" % (link_id + 1, chain_len, link.description_str)

    link.analysis = Analysis(link.scheme, name=link_string_short)

    link.info['t_start'] = time.strftime("%Y/%m/%d %H:%M:%S")
    link.info['t_stamp_start'] = time.time()
    link.info['messages'] = []

    if not searchlight and not verbose == -1:
        print("\n-------  %s  -------" % link.info['t_start'])
        print(link_string)
        print("-------------------------------------\n")

    try:
        link.result = link.analysis.run(n_jobs_folds=n_jobs_folds, verbose=verbose)
    except Exception as ex:
        link.info['success'] = False
        exeception_msg = "An exception of type {0} occured. Arguments:\n{1!r}".\
            format(type(ex).__name__, ex.args)
        link.info['messages'].append(exeception_msg)
        if skip_runerror:
            warn(link.info['messages'][-1] + '\n' + link_string_short)
        else:
            print('[RunErr] ' + link_string_short)
            raise  # re-raise
    finally:
        link.info['success'] = True
        link.info['t_end'] = time.strftime("%Y/%m/%d %H:%M:%S")
        link.info['t_stamp_end'] = time.time()
        link.info['t_dur'] = link.info['t_stamp_end'] - link.info['t_stamp_start']

        if not searchlight and not has_time and link.result:
            detailed_items = ['forest_contrib']
            result = {k: v for k, v in link.result.items() if detailed_save or not
                      any([k.startswith(e) for e in detailed_items])}
            messages = ", ".join(["[" + m + "]" for m in link.info['messages']]) \
                if detailed_save else None
            if detailed_save:
                for k, v in link.description_optional.items():
                    if isinstance(v, np.ndarray):
                        link.description_optional[k] = v.tolist()
                description_optional = json.dumps(link.description_optional) if detailed_save else None
                db_dict = OrderedDict([('scoring', result['scoring']),
                                       ('scoring_ste', result['scoring_ste']),
                                       ('proc_time', link.info['t_dur']),
                                       ('result', json.dumps(result)),
                                       ('t_start', link.info['t_start']),
                                       ('t_end', link.info['t_end']),
                                       ('info', description_optional),
                                       ('timestamp_start', link.info['t_stamp_start']),
                                       ('timestamp_end', link.info['t_stamp_end']),
                                       ('success', link.info['success']),
                                       ('messages', messages)
                                       ] +
                                      list(link.description.items()))
            else:
                db_dict = OrderedDict([('scoring', result['scoring']),
                                       ('scoring_ste', result['scoring_ste']),
                                       ('proc_time', link.info['t_dur']),
                                       ('t_end', link.info['t_end']),
                                       ('correct', json.dumps(result['correct'])),
                                       ] +
                                      list(link.description.items()))
    if not searchlight and not has_time:
        ex_type = None
        try:
            attempts = 0
            # print('[%g %s] Ready to write results' % (link_id, datetime.now().strftime("%H:%M:%S.%f")))
            # if os.path.exists(lockfile):
                # print('[%g %s] Lockfile exists' % (link_id, datetime.now().strftime("%H:%M:%S.%f")))
            while os.path.exists(lockfile):
                # print('[%g %s] Waiting for Lockfile' % (link_id, time.strftime("%H:%M:%S")))
                time.sleep(0.1)
                attempts += 1
                if attempts > 100000:
                    ex_type = 'IOError_LockTimeout'
                    raise IOError("Timeout reached for lock file\n" + link_string)
            # print('[%g %s] Lockfile released' % (link_id, datetime.now().strftime("%H:%M:%S.%f")))
            if os.path.exists(os.path.dirname(lockfile)):
                open(lockfile, 'a').close()
            else:
                os.mkdir(os.path.dirname(lockfile))
            try:
                succeeded = False
                attempts = 0
                while not succeeded:
                    try:
                        # print('[%g %s #%g] Starting insert' % (link_id, datetime.now().strftime("%H:%M:%S.%f"), attempts))
                        # tic = timeit.default_timer()
                        connect(db_string)['chain'].insert(db_dict)
                        # print('[%g %s #%g] Database operation (%.8f secs)' % (link_id, datetime.now().strftime("%H:%M:%S.%f"), attempts, timeit.default_timer()-tic))
                        succeeded = True
                    except Exception as ex:
                        if attempts > 100000:
                            ex_type = 'OperationalError'
                            warning_ = "An exception of type {0} occured. Arguments:\n{1!r}\n".\
                                format(type(ex).__name__, ex.args)
                            warn(warning_ + link_string_short)
                            raise
                        attempts += 1
                        time.sleep(0.1)

            except Exception:
                if os.path.exists(lockfile):
                    os.remove(lockfile)
                if ex_type is None:
                    ex_type = 'IOError'
                raise # re-raise
        except Exception as ex:
            if skip_ioerror:
                warning_ = "An exception of type {0} occured. Arguments:\n{1!r}\n".\
                    format(type(ex).__name__, ex.args)
                warn(warning_ + link_string_short)
            else:
                print('[%s] %s' % (ex_type, link_string_short))
                raise  # re-raise
        finally:
            if os.path.exists(lockfile):
                os.remove(lockfile)
                # print('[%g %s] Lockfile removed' % (link_id, datetime.now().strftime("%H:%M:%S.%f")))

        # total_time = 0
        # while os.path.exists(lockfile):
        #     time.sleep(0.1)
        #     total_time += 0.1
        #     if total_time > 100:
        #         raise IOError("Timeout reached for lock file\n" + link_string)
        # if os.path.exists(os.path.dirname(lockfile)):
        #     open(lockfile, 'a').close()
        # else:
        #     os.mkdir(os.path.dirname(lockfile))
        # # connect(db_string)['chain'].insert(db_dict)
        # attempts = 0
        # succeeded = False
        # while not succeeded:
        #     try:
        #         connect(db_string)['chain'].insert(db_dict)
        #         succeeded = True
        #     except Exception as ex:
        #         if attempts > 5:
        #             warning_ = "An exception of type {0} occured. Arguments:\n{1!r}\n".\
        #                 format(type(ex).__name__, ex.args)
        #             warn(warning_ + link_string_short)
        #             raise
        #         attempts += 1
        #         time.sleep(0.1)
        #
        #
        # if os.path.exists(lockfile):
        #     os.remove(lockfile)


    elif has_time:
        print('Elapsed time: %.1f minutes' % (link.info['t_dur'] / 60.))
        if output_path is not None:
            fname = '%s_%s.pkl' % (link.scheme.data[0].name, link.scheme.channels[0].clfs.name)
            pickle.dump(link.result, open(os.path.join(output_path, fname), 'wb'))
            print('[%s] Saved result in %s' % (time.strftime("%d.%m %H:%M:%S"),
                                               os.path.join(output_path, fname)))

    else:  # searchlight
        for k, img in link.result.items():
            path, ext = os.path.splitext(output_path)
            path_searchlight = '%s_%s%s' % (path, k, ext)
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            if os.path.exists(path_searchlight):
                if ext == '.nii':
                    os.remove(path_searchlight)
            nibabel.save(img, path_searchlight)
            print('[%s] Saved searchlight result in %s' % (time.strftime("%d.%m %H:%M:%S"),
                                                           path_searchlight))
            print('Elapsed time: %.1f minutes' % (link.info['t_dur'] / 60.))

    return link


class ChainBuilder(object):

    scheme_pool = None

    def __init__(self, schemes=None):

        """ Helper to build a decereb.chain.Chain from a (list of ) scheme(s) or scheme name(s)

        Parameters
        ----------
        schemes : str, List<str>, decereb.descriptor.SchemeDescriptor,
                  List<decereb.descriptor.SchemeDescriptor>
        """
        self.schemes = schemes

    @property
    def schemes(self):
        return self._schemes

    @schemes.setter
    def schemes(self, value):
        if not isinstance(value, (list, tuple)):
            value = [value]
        if isinstance(value[0], SchemeDescriptor):
            self.scheme_pool = value
            self._schemes = [v.name for v in value]
        elif isinstance(value[0], str):
            if self.scheme_pool is None:
                raise ValueError('Selection by name requires prior initialization of the scheme '
                                 'pool.')
            else:
                self._schemes = value
        else:
            raise ValueError("Parameter 'schemes' must be one of the following: str, List<str>, "
                             "decereb.descriptor.SchemeDescriptor or "
                             "List<decereb.descriptor.SchemeDescriptor>")

    def build_chain(self, select_indices=None):

        """ Build the chain from provided scheme_names

        Parameters
        ----------
        select_indices: List<int>
            select certain inidices of the chain; practical for e.g. debugging

        Returns
        -------
        chain : decereb.chain.Chain

        """
        selected_schemes = list(DescriptorConcatenator(descriptor_list=self.scheme_pool,
                                                       base_names=self.schemes))

        if select_indices is None:
            select_indices = range(len(selected_schemes))
        elif not isinstance(select_indices, (list, tuple)):
            select_indices = [select_indices]

        link_list = []
        for i, scheme in enumerate(selected_schemes):
            if i in select_indices:
                link_list.append(Link(scheme=scheme))

        chain = Chain(link_list)

        return chain

    @staticmethod
    def init_chain_builder(schemes):
        ChainBuilder.scheme_pool = schemes
