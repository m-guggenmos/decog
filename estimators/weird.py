from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.classification import _weighted_sum
from sklearn.preprocessing.data import robust_scale
from scipy.stats import ttest_ind
import numpy as np
import warnings

WEIGHTING_DICT = {'t': (ttest_ind, {'equal_var': True}), 'welch': (ttest_ind, {'equal_var': False})}
AVERAGING_DICT = {'median': np.median, 'mean': np.mean}


class WEIRD(BaseEstimator, ClassifierMixin):
    """
    WEIRD - weighted robust distance classifier
    __________________________________________________________________________
    Matthias Guggenmos, Katharina Schmack and Philipp Sterzer (2016) WEIRD - a fast and performant
    multivoxel pattern classifier. Proceedings of the 6th International Workshop on Pattern
    Recognition in NeuroImaging: XX-XX, Trento, Italy.

    """

    def __init__(self, centroid_weighting=True, stats_weighting='t', averaging='mean', verbose=0):

        """

        Parameters
        ----------
        centroid_weighting : boolean
            *True*, if votes map the gradual distance from the centroids of the two classes.
            *False*, if votes are binary values indicating to which class the current datapoint is closest.
        stats_weighting : None, str
            *None* switches of statistical weighting of WEIRD
            *'t'* for weighting based on two-sample t-test
            *'welch'* for weighting based on the Welch test
        averaging : str
            *'mean'* centroid computation based on the np.mean operation
            *'median'* centroid computation based on the np.median operation
        verbose : int
            legacy parameter without any function in the present class
        """

        self.centroid_weighting = centroid_weighting
        self.stats_weighting = stats_weighting
        self.averaging = averaging
        self.verbose = verbose

        self.classes_ = None
        self.feature_importances_ = None
        self.averages = None

    def fit(self, X, y):

        """ Train the model.

        Parameters
        ----------
        X : np.ndarray, List
            Data in the form of rows x columns = samples x features
        y : np.ndarray, List
            Class labels, one value per row of X

        Returns
        -------
        The class instance.
        """
        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        x1 = X[np.array(y) == self.classes_[0], :]
        x2 = X[np.array(y) == self.classes_[1], :]

        if self.stats_weighting is not None:
            w = WEIGHTING_DICT[self.stats_weighting]
            with warnings.catch_warnings():  # catching potential warnings about nans
                warnings.simplefilter("ignore", RuntimeWarning)
                statistic = w[0](x1, x2, **w[1]).statistic
            statistic[np.isnan(statistic)] = 0
            self.feature_importances_ = np.atleast_1d(abs(statistic[:, np.newaxis]).squeeze())

        self.averages = np.vstack((AVERAGING_DICT[self.averaging](x1, 0),
                                   AVERAGING_DICT[self.averaging](x2, 0)))

        return self

    def predict(self, X):

        """ Predict new samples based on the trained model.

        Parameters
        ----------
        X : np.ndarray, List
            Data in the form of rows x columns = samples x features

        Returns
        -------
        Predicted class labels.
        """
        dec = self.decision_function(X)
        return self.classes_[(dec > 0).astype(int)]

    def decision_function(self, X):

        """ Compute the (weighted) sum of votes.

        Parameters
        ----------
        X : np.ndarray, List
            Data in the form of rows x columns = samples x features

        Returns
        -------
        The (weighted) sum of votes.

        """
        if self.centroid_weighting:
            votes = robust_scale(abs(X - self.averages[0, :]) - abs(X - self.averages[1, :]),
                                 with_centering=False, axis=1)
        else:
            votes = (abs(X - self.averages[0, :]) > abs(X - self.averages[1, :])) - 0.5
        if self.stats_weighting is None:
            dec = np.sum(votes, 1) / votes.shape[1]
        else:
            dec = _weighted_sum(votes, self.feature_importances_) / votes.shape[1]
        return dec
