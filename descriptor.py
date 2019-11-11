import itertoolsfrom collections import OrderedDictfrom copy import deepcopyfrom inspect import isclass, signaturefrom sklearn.base import BaseEstimatorfrom sklearn.svm import SVCfrom sklearn.model_selection import LeaveOneOutfrom .util.sequences import flattenfrom .util.various import datetime_microsecondsfrom .masker import DummyMasker, MultiRoiMaskerimport numpy as npimport warningsclass Struct(dict):    def __init__(self):        """ Data structure that allows access to attributes both via keys and attributes AND is        compatible with multiprocessing_on_dill.        """        super().__init__()    def __getitem__(self, key):        if key not in self.keys():            raise KeyError('No attribute %s' % key)        return getattr(self, key)    def __setitem__(self, key, value):        super().__setattr__(key, value)        super().__setitem__(key, value)    def __setattr__(self, key, value):        super().__setattr__(key, value)        super().__setitem__(key, value)    def __delitem__(self, *args, **kwargs):        super().__delattr__(*args, **kwargs)        super().__delitem__(*args, **kwargs)    def update(self, *args, **kwargs):        for arg in args:            for key, value in arg:                super().__setattr__(key, value)                super().__setitem__(key, value)        for key, value in kwargs.items():            super().__setattr__(key, value)            super().__setitem__(key, value)    def __iter__(self):        for key in self.keys():            yield key    def keys(self):        return [key for key in super().keys()]    def items(self):        return [(key, value) for key, value in super().items()]    def __repr__(self):        string_ = 'Struct object('        at_least_one_element = False        for key, value in super().items():            string_ += '%s: %s, ' % (key, value)            at_least_one_element = True        if at_least_one_element:            string_ = string_[:-2]        string_ += ')'        return string_class Data(Struct):    def __init__(self, data, name='data', has_time=False,                 feature_names=None, subjects=None, **kwargs):        """ Object to define the data input        Parameters        ----------        data : data            data (e.g. subject x feature matrix, or List<NiftiImages>        name : str            identifying name for the data definition        feature_names : List<str>            provide identifying names for each feature        """        super().__init__()        self.name = name        self.has_time = has_time        self.data = data        self.feature_names = feature_names        self.subjects = subjects        for k, v in kwargs.items():            setattr(self, k, v)    @property    def data(self):        return self._data    @data.setter    def data(self, value):        self._data = valueclass Channel(Struct):    def __init__(self, data=None, masker=None, preproc=None, fss=None, clfs=None):        """ A pipe defines a combination of feature selections and classifiers        Parameters        ----------        data : decereb.descriptor.Data        masker : decereb.descriptor.MaskerDescriptor        preproc : decereb.descriptor.PreprocessingDescriptor        fss : decereb.descriptor.FeatureSelectionDescriptor,              List<decereb.descriptor.FeatureSelectionDescriptor>        clfs : decereb.descriptor.ClassifierDescriptor,               List<decereb.descriptor.ClassifierDescriptor>        """        super().__init__()        self.data = data        self.masker = masker        self.preproc = preproc        self.clfs = self.get_descriptor(clfs)        self.fss = None if fss is None else self.get_descriptor(fss)    def get_descriptor(self, variable):        # if isinstance(variable, (list, tuple)):        #     return list(DescriptorConcatenator(variable)) if len(variable) > 1 else variable[0]        # else:        #     return variable        return list(DescriptorConcatenator(variable))    @property    def masker(self):        return self._masker    @masker.setter    def masker(self, value):        if value is None:            value = MaskerDescriptor(masker=DummyMasker)        self._masker = self.get_descriptor(value)    # def set_clfs(self, value):    #     clf_list = value if isinstance(value, (list, tuple)) else [value]    #     if isinstance(clf_list[0], ClassifierDescriptor):    #         Channel.clf_pool = clf_list    #         clf_list = [v.name for v in clf_list]    #     elif isinstance(value[0], str):    #         if self.clf_pool is None:    #             raise ValueError('Selection by name requires prior initialization of the '    #                              'classifier pool.')    #         else:    #             clf_list = value    #     self.clfs = list(DescriptorConcatenator(self.clf_pool, clf_list))    #    # def set_fss(self, value):    #     if value is None:    #         self.fss = None    #     else:    #         fs_list = value if isinstance(value, (list, tuple)) else [value]    #         if isinstance(fs_list[0], FeatureSelectionDescriptor):    #             Channel.fs_pool = fs_list    #             fs_list = [v.name for v in fs_list]    #         elif isinstance(value[0], str):    #             if self.fs_pool is None:    #                 raise ValueError('Selection by name requires prior initialization of the '    #                                  'feature selection pool.')    #             else:    #                 fs_list = value    #         self.fss = list(DescriptorConcatenator(self.fs_pool, fs_list))    @staticmethod    def init_channel(clfs=None, fss=None):        """ Initialize channel.        Parameters        ----------        clfs : List<decereb.descriptor.ClassifierDescriptor>        fss : List<decereb.descriptor.FeatureSelectionDescriptor>        """        Channel.clf_pool = clfs        Channel.fs_pool = fssclass DescriptorConcatenator:    def __init__(self, descriptor_list, con_names=None):        """        Parameters        ----------        descriptor_list : List<decereb.descriptor.[ClassDerivingFromMetaDescriptor]>        con_names : List<str>            [TODO: is this still used?]        """        self.base_descriptors = sum([list(d) for d in descriptor_list], []) if isinstance(descriptor_list, (list, tuple)) \            else list(descriptor_list)        if con_names is not None:            self.connections = sum(sum([[list(d) for d in self.base_descriptors if d.name == k]                                        for k in con_names], []), [])        else:            self.connections = None    def __iter__(self):        """ Iteration over the DescriptorConcatenator returns all combinations of different        descriptor types.        """        if self.connections is not None:            n = len(self.connections)            perms = [c for i in range(n + 1) for c in itertools.combinations(range(n), i)]            lst = sum([[tuple([deepcopy(base)] + list(deepcopy(con)))                        for con in itertools.product(*[self.connections[i] for i in p])]                       for p in perms for base in self.base_descriptors], [])            for l in lst:                if len(l) > 1:                    for i, descriptor in enumerate(l):                        new_identifier = OrderedDict()                        for k in descriptor.identifier.keys():                            new_key = '%s%g_%s' % \                                      (descriptor._prefix, i, k.split(descriptor._prefix + '_', 1)[1])                            new_identifier[new_key] = descriptor.identifier[k]                        descriptor.identifier = new_identifier        else:            lst = self.base_descriptors        for i, descriptor in enumerate(lst):            if isinstance(descriptor, (list, tuple)):                for d in descriptor:                    d.identifier['%s_id' % d._prefix] = i            else:                descriptor.identifier['%s_id' % descriptor._prefix] = i        for l in lst:            yield lclass Descriptor:    def __init__(self):        """ Parent class for all Descriptor objects        """        self.identifier_exceptions = []        self.iterator_exceptions = []        self.iterator_list = []        self.identifier = None    def __iter__(self):        """ Iteration over the Descriptor returns all combinations of Descriptor        definitions that were provided as Lists.        """        list_args = sum([[(i, k) for k, v in iterator.items()                          if isinstance(v, list) and k not in self.iterator_exceptions]                         for i, iterator in enumerate(self.iterator_list)], [])        if True in [len(la) > 0 for la in list_args]:            instances = []            perms = [list(p) for p in                     list(itertools.product(*[[(i, la, j) for j in self.iterator_list[i][la]]                                              for i, la in list_args]))]            for perm in perms:                instance = deepcopy(self)                for p in perm:                    instance.iterator_list[p[0]].update([p[1:]])                instance.build_identifier()                instances.append(instance)        else:            instances = [self]        for instance in instances:            yield instance    def build_identifier(self):        """ Build identifier, which is a list of (key, value), which is used to generate a        description for the Descriptor (e.g. for database storage)        """        identifiers = []        for identifier in self.identifier_list:            if issubclass(identifier.__class__, dict):                identifiers += [(self._prefix + '_' + k, v) for k, v in identifier.items()                                if k not in self.identifier_exceptions]            elif isinstance(identifier, (list, tuple)):                for i, identifier_ in enumerate(identifier):                    number_prefix = '' if len(identifier) == 1 else str(i) + '_'                    identifiers += [(number_prefix + self._prefix + '_' + k, v)                                    for k, v in identifier_.items()                                    if k not in self.identifier_exceptions]        self.identifier = OrderedDict(identifiers)        for k, v in self.identifier.items():            if isinstance(v, (list, dict)):                self.identifier[k] = str(v)            elif isclass(v):                self.identifier[k] = str(v).split('.')[-1][:-2]            elif hasattr(v, '__class__') and issubclass(v.__class__, BaseEstimator):                self.identifier[k] = str(v)class SchemeDescriptor(Descriptor):    optional_info_items = ['labels', 'subjects', 'label_names']    def __init__(self, channels, labels, name='default', subjects=None, label_names=None,                 meta_clf=None, meta_clf_args=None, cv=None, cv_grid=LeaveOneOut(),                 regression=False, searchlight=False, scoring=None, grid_scoring=None,                 seed_list=None, compute_permutation_importance=False, **kwargs):        """ Definition of a data processing scheme.        Parameters        ----------        data : decereb.descriptor.Data            Data on which the scheme is based        labels : List<int>            provide one label per subject. In a patient/control case, patient should be 1, control            should be 0.        subjects : List<str, int>            provide identifying information for each subject        label_names : List<str>            provide identifying names for each integer label        name : str            Identifying name for the scheme        channels : decereb.descriptor.Channel,  List<decereb.descriptor.Channel>            (List of) processing channels(s)        cv : object            cross validation object: a generator returning (train_indices, test_indices)                                     in each iteration        scoring : str        seed_list : List<int>, None            List of seeds that should be looped over        """        super().__init__()        self.name = name        self._prefix = 'in'        self.labels = np.array(labels)        self.subjects = subjects        self.label_names = label_names        self.channels = channels        self.meta_clf = meta_clf        self.meta_clf_args = meta_clf_args        self.cv = cv        self.cv_grid = cv_grid        self.scoring = scoring        self.grid_scoring = grid_scoring        self.compute_permutation_importance = compute_permutation_importance        self.searchlight = searchlight        self.regression = regression        self.seed_list = seed_list        for k, v in kwargs.items():            setattr(self, k, v)        # make sure feature selection is always a tuple        # for channel in self.channels.values():        #     if channel.fss:        #         if isinstance(channel.fss, list):        #             for f, fs in enumerate(channel.fss):        #                 if not isinstance(fs, tuple):        #                     channel.fss[f] = (fs,)        #         elif not isinstance(channel.fss, tuple):        #             channel.fss = (channel.fss,)        self.iterator_list = [self.channels]        # self.iterator_exceptions = []        self.identifier_list = [dict(name=name)]        # self.identifier_exceptions = []        self.build_identifier()    @property    def channels(self):        return self._channels    @channels.setter    def channels(self, value):        if isinstance(value, Struct):            self._channels = dict(data=value)        else:            self._channels = value        have_time = [v.data.has_time for v in self._channels.values()]        if len(set(have_time)) == 1:            self.has_time = have_time[0]        else:            raise ValueError('Mix of data with and without time information not allowed.')        if not self.has_time:            for channel in self._channels.values():                if len(channel.data.data) != len(self.labels):                    raise ValueError('First dimension of data input must match length of labels!')    @property    def seed_list(self):        return self._seed_list    @seed_list.setter    def seed_list(self, value):        first_time_called = not hasattr(self, '_seed_list')        self._seed_list = value if isinstance(value, (list, tuple, range)) else [value]        self.n_seeds = None if self._seed_list is [None] else len(self._seed_list)        if not first_time_called:            self.build_identifier()    # def __iter__(self):    #     """ Jesus..    #     """    #     list_args = flatten([[(i, k) for k, v in iterator.items()    #                            if isinstance(v, list) and k not in self.iterator_exceptions]    #                          for i, iterator in enumerate(self.iterator_list)], seq_type=list)    #     if True in [len(la) > 0 for la in list_args]:    #         instances = []    #         perms = [list(p) for p in list(itertools.product(*[[(i, la, i)    #                  for i in self.iterator_list[i][la]] for i, la in list_args]))]    #         for perm in perms:    #             perm_names = [p[1] for p in perm]    #             valid = True    #             if 'fs' in perm_names \    #                     and [fs.name for fs in perm[perm_names.index('fs')][-1]] != [None]:    #                 incomp = perm[perm_names.index('clf')][-1].fs_incompatible    #                 fs_list = [fs.name for fs in perm[perm_names.index('fs')][-1]]    #                 if incomp is True \    #                         or (isinstance(incomp, (tuple, list)) and bool(set(fs_list) & set(incomp))):    #                     valid = False    #             if valid:    #                 instance = deepcopy(self)    #                 for p in perm:    #                     instance.iterator_list[p[0]].update([p[1:]])    #                 instance.build_identifier()    #                 instances.append(instance)    #     else:    #         instances = [self]    #    #     for instance in instances:    #         yield instance    def __iter__(self):        """ Jesus..        """        list_args = flatten([[[(i, c, k) for k, v in channel.items()                               if isinstance(v, list) and k not in self.iterator_exceptions]                              for c, channel in iterator.items()]                             for i, iterator in enumerate(self.iterator_list)], seq_type=list)        if True in [len(la) > 0 for la in list_args]:            instances = []            perms = [list(p) for p in list(itertools.product(*[[(i, c, la, j)                     for j in self.iterator_list[i][c][la]] for i, c, la in list_args]))]            for perm in perms:                perm_names = [p[2] for p in perm]                valid = True                if 'fs' in perm_names \                        and [fs.name for fs in perm[perm_names.index('fs')][-1]] != [None]:                    incomp = perm[perm_names.index('clf')][-1].fs_incompatible                    fs_list = [fs.name for fs in perm[perm_names.index('fs')][-1]]                    if incomp is True \                            or (isinstance(incomp, (tuple, list)) and bool(set(fs_list) & set(incomp))):                        valid = False                if valid:                    instance = deepcopy(self)                    for p in perm:                        instance.iterator_list[p[0]][p[1]].update([p[2:]])                    instance.build_identifier()                    instances.append(instance)        else:            instances = [self]        for instance in instances:            yield instanceclass MaskerDescriptor(Descriptor):    def __init__(self, name='masker', masker=None, masker_args=None, **kwargs):        """ Definition of the masker        Parameters        ----------        name : str            Identifying name for the scheme        masker : (List of) object(s) deriving from nilearn.input_data.base_masker.BaseMasker        masker_args : (List of) dict            Arguments to be passed to the masker        """        super().__init__()        self.name = name        self._prefix = 'in'        self.masker = DummyMasker if masker is None else masker        self.masker_args = dict() if masker_args is None else masker_args        if 'rois' in self.masker_args and not isinstance(self.masker_args['rois'], (list, tuple)):            self.masker_args['rois'] = [self.masker_args['rois']]        self.iterator_list = [self.masker_args]        self.iterator_exceptions = ['rois']        self.identifier_list = [dict(name=name),                                self.masker_args]        self.identifier_exceptions = ['rois', 'mask_img']        self.build_identifier()class FeatureSelectionDescriptor(Descriptor):    def __init__(self, fs, name='fs', fs_args=None):        """ Definition of the feature selection        Parameters        ----------        name : None, str            Identifying name for the feature selection descriptor        fs_args :            Arguments for the feature selection (as defined by the name)        """        super().__init__()        self.fs = fs        self.name = name        self.fs_args = fs_args        self._prefix = 'fs'        self.identifier_exceptions = ['model_args']        self._build_identifier_list()        self.build_identifier()    @property    def fs_args(self):        return self._fs_args    @fs_args.setter    def fs_args(self, value):        first_time_called = not hasattr(self, '_fs_args')        if value is None:            self._fs_args = dict()        else:            self._fs_args = value            self.iterator_list += [self._fs_args]            if 'model_args' in self._fs_args:                self.iterator_list += [self.fs_args['model_args']]        if not first_time_called:            self._build_identifier_list()            self.build_identifier()    def _build_identifier_list(self):        self.identifier_list = [dict(name=self.name), *self.iterator_list]class ClassifierDescriptor(Descriptor):    def __init__(self, name, clf=SVC, clf_args=None, fs_incompatible=('',),                 collect_variables=None):        """ Definition of the classifier        Parameters        ----------        name : str            Identifying name for the classifier descriptor        clf : classifier class            Uninitialized class of the classifier (e.g. sklearn.svm.LinearSVC)        clf_args : dict            Arguments to be passed to the classifier        fs_incompatible : str, Tuple<str>            (Tuple of) String(s) that specifies incompatible feature selection descriptors by name        """        super().__init__()        self._clf_string = str(clf)[8:-2]        self.clf_args = clf_args        self.name = name        self.fs_incompatible = fs_incompatible        self.collect_variables = collect_variables        self.clf = clf        self._prefix = 'clf'        self.iterator_exceptions = ['seed_list']        self.identifier_exceptions = ['seed_list']        self._build_identifier_list()        self.build_identifier()    @property    def name(self):        return self._name    @name.setter    def name(self, value):        if value is None:            self._name = self._clf_string + '_' + datetime_microseconds()        else:            self._name = value    @property    def collect_variables(self):        return self._collect_variables    @collect_variables.setter    def collect_variables(self, value):        if isinstance(value, list) or value is None:            self._collect_variables = value        elif isinstance(value, str):            self._collect_variables = [value]        else:            raise ValueError('collect_variables must be None, string or list')    @property    def clf(self):        return self._clf    @clf.setter    def clf(self, value):        first_time_called = not hasattr(self, '_clf')        self._clf = value        if not first_time_called:            self._build_identifier_list()            self.build_identifier()    @property    def clf_args(self):        return self._clf_args    @clf_args.setter    def clf_args(self, value):        first_time_called = not hasattr(self, '_clf_args')        if value is None:            self._clf_args = dict()        else:            self._clf_args = value            self.iterator_list += [self._clf_args]        if not first_time_called:            self._build_identifier_list()            self.build_identifier()    @property    def fs_incompatible(self):        return self._fs_incompatible    @fs_incompatible.setter    def fs_incompatible(self, value):        if isinstance(value, (list, tuple)):            self._fs_incompatible = value        else:            self._fs_incompatible = (value, )    def _build_identifier_list(self):        self.identifier_list = [dict(name=self.name, clf=self._clf_string),                                *self.iterator_list]class PreprocessingDescriptor(Descriptor):    def __init__(self, name='preproc', preprocessor=None, preprocessor_args=None):        super().__init__()        self.name = name        self.preprocessor = preprocessor        self.preprocessor_args = preprocessor_args        # self.build_identifier()