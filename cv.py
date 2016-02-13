import numpy as np


class XClassSplit:

    def __init__(self, sets, runs):
        self.sets = np.array(sets, dtype=np.int)
        self.runs = np.array(runs, dtype=np.int)

        self.unique_runs = np.unique(self.runs)
        self.unique_sets = np.unique(self.sets)
        self.n = len(self.unique_sets) * len(self.unique_runs)

    def __iter__(self):
        for set in self.unique_sets:
            for run in self.unique_runs:
                test_index = np.where((self.sets == set) & (self.runs == run))[0]
                train_index = np.where((self.sets != set) & (self.runs != run))[0]
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

    sets = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    runs = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    cv = XClassSplit(runs=runs, sets=sets)

    for train, test in cv:
        print("TRAIN:", train, "TEST:", test)
        y_train, y_test = y[train], y[test]
        print("y_TRAIN:", y_train, "y_TEST:", y_test)
