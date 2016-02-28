from collections import OrderedDict, Sequence
import time
from dataset import connect
import os
import multiprocessing_on_dill as multiprocessing
import json
from warnings import warn
import nibabel

from .util.archiving import zip_directory_structure
from .analysis import Analysis
from .descriptor import DescriptorConcatenator, Data


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
                        for k, v in channel.clf.identifier.items()]
                       for c, channel in enumerate(self.scheme.channels)]
        fs_descrip = [[[('%s%s' % (prefix(c, is_multichannel), k), v)
                        for k, v in fs.identifier.items()] for fs in channel.fs]
                      if channel.fs is not None
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

    def run(self, n_jobs_links=1, n_jobs_folds=1, verbose=2, seed=0, output_path=None,
            recompute=False, skip_runerror=True, skip_ioerror=False, zip_code_dirs=None,
            detailed_save=False):
        """ Start the analyses chain
        Parameters
        ----------
        n_jobs_links:  int
            number of parallell jobs for links
        n_jobs_folds: int
            number of parallell jobs for folds
        verbose: int
            verbosity level
        seed : int, None [TODO]
            seed for random number generator (None = no seed is used)
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

        if not self.link_list[0].scheme.channels[0].clf.searchlight:
            if output_path is None:
                db_string = 'sqlite:///:memory:'
                output_dir = '/tmp/'
            elif os.path.splitext(output_path)[1] == '.db':
                output_dir = os.path.splitext(output_path)[0]
                if os.path.exists(output_path) and not recompute:
                    db_string = 'sqlite:///%s' % output_path
                    db = connect(db_string)
                    in_db = []
                    for i, link in enumerate(self.link_list):
                        where = ' '.join(["AND %s IS '%s'" % (k, v)
                                          for k, v in link.description.items()])[4:]
                        if list(db.query('SELECT id FROM chain WHERE %s' % where)):
                            in_db.append(i)
                    for i in sorted(in_db, reverse=True):
                        print('deleting %s' % self.link_list[i].description_str)
                        del self.link_list[i] # if entry in database, remove from linkdef_list
                else:
                    db_string = 'sqlite:///%s_%s.db' % (os.path.splitext(output_path)[0], timestamp)
            else:
                output_dir = output_path
                db_string = 'sqlite:///%s' % os.path.join(output_path, 'data_%s.db' % timestamp)
        else:
            db_string = None
            output_dir = os.path.dirname(output_path)

        if zip_code_dirs is not None:
            zip_path = os.path.join(output_dir, 'archive_%s.zip' % timestamp)
            zip_directory_structure(zip_code_dirs, zip_path, allowed_pattern='*.py')

        lockfile = os.path.join(output_dir, 'lock_%s' % timestamp)

        if n_jobs_links == 1:
            chain = []
            for link_id, link in enumerate(self.link_list):
                params = (n_jobs_folds, verbose, seed, link_id, len(self.link_list),
                          link, skip_runerror, skip_ioerror, db_string, lockfile, output_path,
                          timestamp, detailed_save)
                link = _link(params)
                chain.append(link)
        else:
            pool = multiprocessing.Pool(None if n_jobs_links == -1 else n_jobs_links)
            params = [(n_jobs_folds, verbose, seed, link_id, len(self.link_list),
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

        return chain


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
    n_jobs_folds, verbose, seed, link_id, chain_len, link, skip_runerror, \
        skip_ioerror, db_string, lockfile, output_path, timestamp, detailed_save = params

    searchlight = link.scheme.channels[0].clf.searchlight

    link_string = "Running chain link %g / %g %s\n%s" % \
                  (link_id + 1, chain_len, db_string, link.description_str)
    link_string_short = "[%g/%g] %s" % (link_id + 1, chain_len, link.description_str)

    link.analysis = Analysis(link.scheme, name=link_string_short)

    link.info['t_start'] = time.strftime("%Y/%m/%d %H:%M:%S")
    link.info['t_stamp_start'] = time.time()
    link.info['messages'] = []

    if not searchlight:
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

        if not searchlight and link.result:
            detailed_items = ['forest_contrib']
            result = {k: v for k, v in link.result.items() if detailed_save or not
                      any([k.startswith(e) for e in detailed_items])}
            messages = ", ".join(["[" + m + "]" for m in link.info['messages']]) \
                if detailed_save else None
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
    if not searchlight:
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
                warning_ = "An exception of type {0} occured. Arguments:\n{1!r}\n".\
                    format(type(ex).__name__, ex.args)
                warn(warning_ +  + link_string_short)
            else:
                print('[IOErr] ' + link_string_short)
                raise  # re-raise
        finally:
            if os.path.exists(lockfile):
                os.remove(lockfile)
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

    schemes = None

    def __init__(self, scheme_names=None):

        """ Helper to build a decereb.chain.Chain from a (list of )scheme name(s)

        Parameters
        ----------
        scheme_names : str, List<str>
        """
        self.scheme_names = scheme_names

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
        selected_schemes = list(DescriptorConcatenator(descriptor_list=self.schemes,
                                                       base_names=self.scheme_names))

        if select_indices is None:
            select_indices = range(len(selected_schemes))
        elif not isinstance(select_indices, Sequence):
            select_indices = [select_indices]

        link_list = []
        for i, scheme in zip(select_indices, selected_schemes):
            link_list.append(Link(scheme=scheme))

        chain = Chain(link_list)

        return chain

    @staticmethod
    def init_chain_builder(schemes):
        ChainBuilder.schemes = schemes
