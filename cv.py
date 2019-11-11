import numpy as np
from itertools import product, combinations
from scipy.special import binom

class ValidationSplit():
    def __init__(self, labels_train, labels_test):

        self.labels_train = labels_train
        self.labels_test = labels_test

    def __iter__(self):
        yield np.arange(len(self.labels_train)), \
              np.arange(len(self.labels_train), len(self.labels_train) + len(self.labels_test))

    def split(self, X, y=None):
        return self.__iter__()

    def __len__(self):
        return 1

    def get_n_splits(self, X=None, y=None):
        return 1


class ShuffleBinLeaveOneOut:

    def __init__(self, labels, n_iter=10, n_pseudo=5):

        """

        Parameters
        ----------
        labels : List<int>, np.ndarray()
            Label for each trial
        n_iter : int
            Number of permutations
        n_pseudo : int
            How many trials belong to one bin (aka pseudo-trial)
        """

        self.labels = np.array(labels)
        self.n_iter = n_iter

        self.classes, self.n_trials = np.unique(labels, return_counts=True)
        self.n_classes = self.classes.shape[0]
        self.n_pseudo = n_pseudo
        self._compute_pseudo_info()

    def _compute_pseudo_info(self):
        """
        Compute indices and labels for the pseudo-trial matrix
        The pseudo-trial matrix is the resulting matrix *after* having grouped the data into
        randomly permuted pseudo-trial bins and averaging trials within each bin. Thus, no
        additional permutation is necessary for pseudo-trial indices, which are somewhat trivial.
        """
        self.ind_pseudo_train = np.full((self.n_classes, self.n_classes, 2*(self.n_pseudo-1)),
                                        np.nan, dtype=np.int)
        self.ind_pseudo_test = np.full((self.n_classes, self.n_classes, 2), np.nan, dtype=np.int)
        self.labels_pseudo_train = np.full((self.n_classes, self.n_classes, 2*(self.n_pseudo-1)),
                                           np.nan, dtype=np.int)
        self.labels_pseudo_test = np.full((self.n_classes, self.n_classes, 2), np.nan, dtype=np.int)
        for c1 in range(self.n_classes):
            range_c1 = range(c1*(self.n_pseudo-1), (c1+1)*(self.n_pseudo-1))
            for c2 in range(self.n_classes):
                range_c2 = range(c2*(self.n_pseudo-1), (c2+1)*(self.n_pseudo-1))
                self.ind_pseudo_train[c1, c2, :2*(self.n_pseudo - 1)] = \
                    np.concatenate((range_c1, range_c2))
                self.ind_pseudo_test[c1, c2] = [c1, c2]

                self.labels_pseudo_train[c1, c2, :2*(self.n_pseudo - 1)] = \
                    np.concatenate((self.classes[c1] * np.ones(self.n_pseudo - 1),
                                    self.classes[c2] * np.ones(self.n_pseudo - 1)))
                self.labels_pseudo_test[c1, c2] = self.classes[[c1, c2]].astype(self.labels_pseudo_train.dtype)

    def __iter__(self):

        """
        Generator function for the cross-validation object. Each fold corresponds to a new random
        grouping of trials into pseudo-trials.
        """
        _ind_train = np.full(self.n_classes*(self.n_pseudo-1), np.nan, dtype=np.object)
        _ind_test = np.full(self.n_classes, np.nan, dtype=np.object)
        for perm in range(self.n_iter):
            for c1 in range(self.n_classes):  # separate permutation for each class
                prm = np.array(np.array_split(np.random.permutation(self.n_trials[c1]), self.n_pseudo))
                ind = prm + np.sum(self.n_trials[:c1])
                for i, j in enumerate(range(c1*(self.n_pseudo-1), (c1+1)*(self.n_pseudo-1))):
                    _ind_train[j] = ind[i]
                _ind_test[c1] = ind[-1]
            yield _ind_train, _ind_test

    def split(self, X, y=None):
        return self.__iter__()

    def get_n_splits(self, X=None, y=None):
        return self.n_iter

    def __len__(self):
        return self.n_iter

class ShuffleBinLeaveOneOutWithin:

    def __init__(self, labels, n_iter=10, n_pseudo=5):

        """

        Parameters
        ----------
        labels : List<int>, np.ndarray()
            Label for each trial
        n_iter : int
            Number of permutations
        n_pseudo : int
            How many trials belong to one bin (aka pseudo-trial)
        """

        self.labels = np.array(labels)
        self.n_iter = n_iter

        self.classes, self.n_trials = np.unique(labels, return_counts=True)
        self.n_classes = self.classes.shape[0]
        self.n_pseudo = n_pseudo
        self._compute_pseudo_info()

    def _compute_pseudo_info(self):
        """
        Compute indices and labels for the pseudo-trial matrix
        The pseudo-trial matrix is the resulting matrix *after* having grouped the data into
        randomly permuted pseudo-trial bins and averaging trials within each bin. Thus, no
        additional permutation is necessary for pseudo-trial indices, which are somewhat trivial.
        """
        self.ind_pseudo_train = np.full((self.n_classes, 1, self.n_pseudo-2),
                                        np.nan, dtype=np.int)
        self.ind_pseudo_test = np.full((self.n_classes, 1, 2), np.nan, dtype=np.int)
        self.labels_pseudo_train = np.full((self.n_classes, 1, self.n_pseudo-2),
                                           np.nan, dtype=np.int)
        self.labels_pseudo_test = np.full((self.n_classes, 1, 2), np.nan, dtype=np.int)
        for c1 in range(self.n_classes):
            range_c1 = range(c1*(self.n_pseudo-2), (c1+1)*(self.n_pseudo-2))

            self.ind_pseudo_train[c1, 0, :self.n_pseudo - 2] = range_c1
            self.ind_pseudo_test[c1, 0] = [c1*2, c1*2+1]

            self.labels_pseudo_train[c1, 0, :self.n_pseudo - 2] = self.classes[c1] * np.ones(self.n_pseudo - 2)
            self.labels_pseudo_test[c1, 0] = self.classes[[c1, c1]].astype(self.labels_pseudo_train.dtype)

    def __iter__(self):

        """
        Generator function for the cross-validation object. Each fold corresponds to a new random
        grouping of trials into pseudo-trials.
        """
        _ind_train = np.full(self.n_classes*(self.n_pseudo-2), np.nan, dtype=np.object)
        _ind_test = np.full(self.n_classes*2, np.nan, dtype=np.object)
        for perm in range(self.n_iter):
            for c1 in range(self.n_classes):  # separate permutation for each class
                prm = np.array(np.array_split(np.random.permutation(self.n_trials[c1]), self.n_pseudo))
                ind = prm + np.sum(self.n_trials[:c1])
                for i, j in enumerate(range(c1*(self.n_pseudo-2), (c1+1)*(self.n_pseudo-2))):
                    _ind_train[j] = ind[i]
                for i, j in enumerate(range(c1*2, (c1+1)*2)):
                    _ind_test[j] = ind[-i-1]
            yield _ind_train, _ind_test


    def split(self, X, y=None):
        return self.__iter__()

    def __len__(self):
        return self.n_iter

    def get_n_splits(self, X=None, y=None):
        return self.n_iter

class ShuffleBin:

    def __init__(self, labels, n_iter=10, n_pseudo=5):

        """

        Parameters
        ----------
        labels : List<int>, np.ndarray()
            Label for each trial
        n_iter : int
            Number of permutations
        n_pseudo : int
            How many trials belong to one bin (aka pseudo-trial)
        """

        self.labels = np.array(labels)
        self.n_iter = n_iter

        self.classes, self.n_trials = np.unique(labels, return_counts=True)
        self.n_classes = self.classes.shape[0]
        self.n_pseudo = n_pseudo
        self._compute_pseudo_info()

    def _compute_pseudo_info(self):
        """
        Compute indices and labels for the pseudo-trial matrix
        The pseudo-trial matrix is the resulting matrix *after* having grouped the data into
        randomly permuted pseudo-trial bins and averaging trials within each bin. Thus, no
        additional permutation is necessary for pseudo-trial indices, which are somewhat trivial.
        """
        self.ind_pseudo_test = np.full((self.n_classes, self.n_classes, 2 * self.n_pseudo),
                                       np.nan, dtype=np.int)
        self.labels_pseudo_test = np.full((self.n_classes, self.n_classes, 2 * self.n_pseudo),
                                          np.nan, dtype=np.int)

        for c1 in range(self.n_classes):
            range_c1 = range(c1*self.n_pseudo, (c1+1)*self.n_pseudo)
            for c2 in range(self.n_classes):
                range_c2 = range(c2*self.n_pseudo, (c2+1)*self.n_pseudo)
                self.ind_pseudo_test[c1, c2, :2 * self.n_pseudo] = np.concatenate((range_c1, range_c2))
                self.labels_pseudo_test[c1, c2, :2 * self.n_pseudo] = \
                    np.concatenate((self.classes[c1] * np.ones(self.n_pseudo),
                                    self.classes[c2] * np.ones(self.n_pseudo)))
        self.ind_pseudo_train = []
        self.labels_pseudo_train = []


    def __iter__(self):

        """
        Generator function for the cross-validation object. Each fold corresponds to a new random
        grouping of trials into pseudo-trials.
        """
        _ind_test = np.full(self.n_classes*self.n_pseudo, np.nan, dtype=np.object)
        _ind_train = []
        for perm in range(self.n_iter):
            for c1 in range(self.n_classes):  # separate permutation for each class
                prm = np.array(np.array_split(np.random.permutation(self.n_trials[c1]), self.n_pseudo))
                ind = prm + np.sum(self.n_trials[:c1])
                for i, j in enumerate(range(c1*self.n_pseudo, (c1+1)*self.n_pseudo)):
                    _ind_test[j] = ind[i]
            yield _ind_train, _ind_test


    def split(self, X, y=None):
        return self.__iter__()

    def __len__(self):
        return self.n_iter

    def get_n_splits(self, X=None, y=None):
        return self.n_iter


class ShuffleBinTest:

    def __init__(self, labels, n_iter=10, n_pseudo=5):

        """

        Parameters
        ----------
        labels : List<int>, np.ndarray()
            Label for each trial
        n_iter : int
            Number of permutations
        n_pseudo : int
            How many trials belong to one bin (aka pseudo-trial)
        """

        self.labels = np.array(labels)
        self.n_iter = n_iter

        self.classes, self.n_trials = np.unique(labels, return_counts=True)
        self.n_classes = self.classes.shape[0]
        self.n_pseudo = n_pseudo
        self._compute_pseudo_info()

    def _compute_pseudo_info(self):
        """
        Compute indices and labels for the pseudo-trial matrix
        The pseudo-trial matrix is the resulting matrix *after* having grouped the data into
        randomly permuted pseudo-trial bins and averaging trials within each bin. Thus, no
        additional permutation is necessary for pseudo-trial indices, which are somewhat trivial.
        """
        self.ind_pseudo_test = np.full((self.n_classes, self.n_classes, 2 * self.n_pseudo),
                                       np.nan, dtype=np.int)
        self.labels_pseudo_test = np.full((self.n_classes, self.n_classes, 2 * self.n_pseudo),
                                          np.nan, dtype=np.int)

        for c1 in range(self.n_classes):
            range_c1 = range(c1*self.n_pseudo, (c1+1)*self.n_pseudo)
            for c2 in range(self.n_classes):
                range_c2 = range(c2*self.n_pseudo, (c2+1)*self.n_pseudo)
                self.ind_pseudo_test[c1, c2, :2 * self.n_pseudo] = np.concatenate((range_c1, range_c2))
                self.labels_pseudo_test[c1, c2, :2 * self.n_pseudo] = \
                    np.concatenate((self.classes[c1] * np.ones(self.n_pseudo),
                                    self.classes[c2] * np.ones(self.n_pseudo)))
        self.ind_pseudo_train = []
        self.labels_pseudo_train = []


    def __iter__(self):

        """
        Generator function for the cross-validation object. Each fold corresponds to a new random
        grouping of trials into pseudo-trials.
        """
        _ind_test = np.full(self.n_classes*self.n_pseudo, np.nan, dtype=np.object)
        _ind_train = []
        for perm in range(self.n_iter):
            for c1 in range(self.n_classes):  # separate permutation for each class
                prm = np.array(np.array_split(range(self.n_trials[c1]), self.n_pseudo))
                ind = prm + np.sum(self.n_trials[:c1])
                for i, j in enumerate(range(c1*self.n_pseudo, (c1+1)*self.n_pseudo)):
                    _ind_test[j] = ind[i]
            yield _ind_train, _ind_test


    def split(self, X, y=None):
        return self.__iter__()

    def __len__(self):
        return self.n_iter

    def get_n_splits(self):
        return self.n_iter


class ShuffleBinWithin:

    def __init__(self, labels, n_iter=10, n_pseudo=5):

        """

        Parameters
        ----------
        labels : List<int>, np.ndarray()
            Label for each trial
        n_iter : int
            Number of permutations
        n_pseudo : int
            How many trials belong to one bin (aka pseudo-trial)
        """

        self.labels = np.array(labels)
        self.n_iter = n_iter

        self.classes, self.n_trials = np.unique(labels, return_counts=True)
        self.n_classes = self.classes.shape[0]
        self.n_pseudo = n_pseudo
        self._compute_pseudo_info()

    def _compute_pseudo_info(self):
        """
        Compute indices and labels for the pseudo-trial matrix
        The pseudo-trial matrix is the resulting matrix *after* having grouped the data into
        randomly permuted pseudo-trial bins and averaging trials within each bin. Thus, no
        additional permutation is necessary for pseudo-trial indices, which are somewhat trivial.
        """
        self.ind_pseudo_test = np.full((self.n_classes, 1, self.n_pseudo),
                                       np.nan, dtype=np.int)
        self.labels_pseudo_test = np.full((self.n_classes, 1, self.n_pseudo),
                                          np.nan, dtype=np.int)
        for c1 in range(self.n_classes):
            range_c1 = range(c1*self.n_pseudo, (c1+1)*self.n_pseudo)

            self.ind_pseudo_test[c1, 0, :self.n_pseudo] = range_c1
            self.labels_pseudo_test[c1, 0, :self.n_pseudo] = self.classes[c1] * np.ones(self.n_pseudo)
        self.ind_pseudo_train = []
        self.labels_pseudo_train = []


    def __iter__(self):

        """
        Generator function for the cross-validation object. Each fold corresponds to a new random
        grouping of trials into pseudo-trials.
        """
        _ind_test = np.full(self.n_classes*self.n_pseudo, np.nan, dtype=np.object)
        _ind_train = []
        for perm in range(self.n_iter):
            for c1 in range(self.n_classes):  # separate permutation for each class
                prm = np.array(np.array_split(np.random.permutation(self.n_trials[c1]), self.n_pseudo))
                ind = prm + np.sum(self.n_trials[:c1])
                for i, j in enumerate(range(c1*self.n_pseudo, (c1+1)*self.n_pseudo)):
                    _ind_test[j] = ind[i]
            yield _ind_train, _ind_test

    def split(self, X, y=None):
        return self.__iter__()

    def __len__(self):
        return self.n_iter

    def get_n_splits(self, X=None, y=None):
        return self.n_iter


class ShuffleBinWithinTest:

    def __init__(self, labels, n_iter=10, n_pseudo=5):

        """

        Parameters
        ----------
        labels : List<int>, np.ndarray()
            Label for each trial
        n_iter : int
            Number of permutations
        n_pseudo : int
            How many trials belong to one bin (aka pseudo-trial)
        """

        self.labels = np.array(labels)
        self.n_iter = n_iter

        self.classes, self.n_trials = np.unique(labels, return_counts=True)
        self.n_classes = self.classes.shape[0]
        self.n_pseudo = n_pseudo
        self._compute_pseudo_info()

    def _compute_pseudo_info(self):
        """
        Compute indices and labels for the pseudo-trial matrix
        The pseudo-trial matrix is the resulting matrix *after* having grouped the data into
        randomly permuted pseudo-trial bins and averaging trials within each bin. Thus, no
        additional permutation is necessary for pseudo-trial indices, which are somewhat trivial.
        """
        self.ind_pseudo_test = np.full((self.n_classes, 1, self.n_pseudo),
                                       np.nan, dtype=np.int)
        self.labels_pseudo_test = np.full((self.n_classes, 1, self.n_pseudo),
                                          np.nan, dtype=np.int)
        for c1 in range(self.n_classes):
            range_c1 = range(c1*self.n_pseudo, (c1+1)*self.n_pseudo)

            self.ind_pseudo_test[c1, 0, :self.n_pseudo] = range_c1
            self.labels_pseudo_test[c1, 0, :self.n_pseudo] = self.classes[c1] * np.ones(self.n_pseudo)
        self.ind_pseudo_train = []
        self.labels_pseudo_train = []


    def __iter__(self):

        """
        Generator function for the cross-validation object. Each fold corresponds to a new random
        grouping of trials into pseudo-trials.
        """
        _ind_test = np.full(self.n_classes*self.n_pseudo, np.nan, dtype=np.object)
        _ind_train = []
        for perm in range(self.n_iter):
            for c1 in range(self.n_classes):  # separate permutation for each class
                prm = np.array(np.array_split(range(self.n_trials[c1]), self.n_pseudo))
                ind = prm + np.sum(self.n_trials[:c1])
                for i, j in enumerate(range(c1*self.n_pseudo, (c1+1)*self.n_pseudo)):
                    _ind_test[j] = ind[i]
            yield _ind_train, _ind_test

    def split(self, X, y=None):
        return self.__iter__()

    def __len__(self):
        return self.n_iter

    def get_n_splits(self, X=None, y=None):
        return self.n_iter

class XClassSplit():

    def __init__(self, runs, sets):
        self.sets = np.atleast_2d(sets)
        self.runs = np.array(runs, dtype=np.int)

        self.unique_runs = np.unique(self.runs)
        self.unique_sets = np.atleast_2d([np.unique(s) for s in self.sets])
        self.n = sum([len(s) * len(self.unique_runs) for s in self.unique_sets])

    def __iter__(self):

        for s, set in enumerate(self.sets):
            for set_id in self.unique_sets[s]:
                for run in self.unique_runs:
                    test_index = np.where((set == set_id) & (self.runs == run))[0]
                    train_index = np.where((set != set_id) & (self.runs != run))[0]
                    yield train_index, test_index

    def split(self, X, y=None):
        return self.__iter__()

    def __len__(self):
        return self.n

    def get_n_splits(self, X=None, y=None):
        return self.n


# class ExhaustiveLeave2Out:
#
#     def __init__(self, labels):
#         self.labels = labels
#         self.classes = np.unique(self.labels)
#         self.n_samples = len(self.labels)
#         n_samples1 = np.sum(self.labels == self.classes[0])
#         n_samples2 = np.sum(self.labels == self.classes[1])
#         self.n_iter = 2 * n_samples1 * n_samples2
#
#     def __iter__(self):
#         for i, l in enumerate(self.labels):
#             other_class = self.classes[0] if l == self.classes[1] else self.classes[1]
#             ind_other_class = np.where(self.labels == other_class)[0]
#             for i in ind_other_class:
#                 test_ind = [i, i]
#                 train_ind = np.setdiff1d(range(self.n_samples), test_ind)
#                 yield train_ind, test_ind
#
#     def split(self, X, y=None):
#         return self.__iter__()
#
#     def __len__(self):
#         return self.n_iter
#
#     def get_n_splits(self):
#         return self.n_iter


class SuperExhaustiveLeaveNOut:

    def __init__(self, N):
        self.N = N

    def __iter__(self, y):
        for test_ind in combinations(range(len(y)), self.N):
            train_ind = np.setdiff1d(range(len(y)), test_ind)
            yield train_ind, np.array(test_ind)

    def split(self, X, y, groups=None):
        return self.__iter__(y)

    def get_n_splits(self, X, y, groups=None):
        return int(binom(len(y), self.N))


class ExhaustiveLeave2Out:

    def __init__(self):
        pass

    def __iter__(self, y):
        classes = np.unique(y)
        ind1 = np.where(y == classes[0])[0]
        ind2 = np.where(y == classes[1])[0]
        for test_ind in product(ind1, ind2):
            train_ind = np.setdiff1d(range(len(y)), test_ind)
            yield train_ind, np.array(test_ind)

    def split(self, X, y, groups=None):
        return self.__iter__(y)

    def get_n_splits(self, X, y, groups=None):
        classes = np.unique(y)
        n_samples1 = np.sum(y == classes[0])
        n_samples2 = np.sum(y == classes[1])
        return n_samples1 * n_samples2


class SubsetLeave2Out:

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def __iter__(self, y):
        classes = np.unique(y)
        ind1 = np.where(y == classes[0])[0]
        ind2 = np.where(y == classes[1])[0]
        combos = list(product(ind1, ind2))
        order = np.random.choice(len(combos), self.n_splits, replace=False)
        for i in range(self.n_splits):
            test_ind = np.array(combos[order[i]])
            train_ind = np.setdiff1d(range(len(y)), test_ind)
            yield train_ind, test_ind

    def split(self, X, y, groups=None):
        return self.__iter__(y)

    def get_n_splits(self, X, y=None, groups=None):
        return self.n_splits


class ProxyCV:

    def __init__(self, train_ind, test_ind):
        self.train_ind = train_ind
        self.test_ind = test_ind

    def __iter__(self):
        yield self.train_ind, self.test_ind

    def split(self, X, y=None):
        return self.__iter__()

    def __len__(self):
        return 1

    def get_n_splits(self):
        return 1


class DummyCV:

    def __init__(self, n_samples):
        self.n_samples = n_samples

    def __iter__(self):
        yield list(range(self.n_samples)), []

    def split(self, X, y=None):
        return self.__iter__()

    def __len__(self):
        return 1

    def get_n_splits(self):
        return 1


if __name__ == '__main__':

    y = np.array(['r1_A1', 'r1_A2', 'r1_B1', 'r1_B2','r2_A1', 'r2_A2', 'r2_B1', 'r2_B2','r3_A1', 'r3_A2', 'r3_B1', 'r3_B2'])

    sets = [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]]
    runs = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    cv = XClassSplit(runs, sets)

    for train, test in cv:
        print("TRAIN:", train, "TEST:", test)
        y_train, y_test = y[train], y[test]
        print("y_TRAIN:", y_train, "y_TEST:", y_test)