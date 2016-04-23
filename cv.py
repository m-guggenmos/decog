import numpy as np


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
                self.ind_pseudo_train[c1, c2, :self.n_pseudo + self.n_pseudo - 2] = \
                    np.concatenate((range_c1, range_c2))
                self.ind_pseudo_test[c1, c2] = [c1, c2]

                self.labels_pseudo_train[c1, c2, :self.n_pseudo + self.n_pseudo - 2] = \
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

    def __len__(self):
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

    def __len__(self):
        return self.n

class DummyCV:

    def __init__(self, n_samples):
        self.n_samples = n_samples

    def __iter__(self):
        yield list(range(self.n_samples)), []

    def __len__(self):
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