from abc import ABCMeta

import nibabel
import numpy as np
from nilearn.image import resample_img
from scipy.stats import mode
from sklearn import neighbors
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble.base import BaseEnsemble
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
from sklearn.svm import LinearSVC

from decog.estimators.searchlight import SearchLight


class SearchlightEnsemble(six.with_metaclass(ABCMeta, BaseEnsemble)):

    _estimator_type = "searchlight_ensemble"

    def __init__(self, mask_img=None, process_mask_img=None, radius=4, base_estimator=LinearSVC(), n_estimators=5,
                 base_estimator_args=None, verbose=0, n_jobs=1):

        if base_estimator_args is None:
            base_estimator_args = dict()
        super(SearchlightEnsemble, self).__init__(base_estimator, n_estimators=n_estimators,
                                                  estimator_params=tuple(base_estimator_args.keys()))

        for k, v in base_estimator_args.items():
            self.__setattr__(k, v)

        self.mask_img = mask_img
        self.process_mask_img = process_mask_img
        self.radius = radius
        self.base_estimator_ = base_estimator
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.base_estimator_args = base_estimator_args
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.estimators_ = []
        self.best_spheres = None

    def fit(self, X, y):
        """Fit estimators from the training set (X, y).

        Returns
        -------
        self : object
            Returns self.
        """

        if self.base_estimator_._estimator_type == 'classifier':
            scoring = None
        else:
            scoring = 'mean_squared_error'

        X_test = nibabel.load(X[0]) if isinstance(X[0], str) else X[0]
        mask_test = nibabel.load(self.mask_img) if isinstance(self.mask_img, str) else self.mask_img
        process_mask_test = nibabel.load(self.process_mask_img) if isinstance(self.process_mask_img, str) else self.process_mask_img
        if not np.array_equal(X_test.affine, mask_test.affine) or not np.array_equal(X_test.shape, mask_test.shape):
            self.mask_img = resample_img(mask_test, target_affine=X_test.affine, target_shape=X_test.shape, interpolation='nearest')
        if not np.array_equal(X_test.affine, process_mask_test.affine) or not np.array_equal(X_test.shape, process_mask_test.shape):
            self.process_mask_img = resample_img(process_mask_test, target_affine=X_test.affine, target_shape=X_test.shape, interpolation='nearest')

        searchlight = SearchLight(self.mask_img, process_mask_img=self.process_mask_img, estimator=self.base_estimator_, scoring=scoring,
                                  radius=self.radius, n_jobs=self.n_jobs, estimator_params=self.base_estimator_args, cv=LeaveOneOut())
        searchlight.fit(X, y)
        if np.all(searchlight.scores_ == 0):
            raise RuntimeError('Ooops, something went probably wrong: all searchlight scores have value 0.')
        if self.base_estimator_._estimator_type == 'classifier':
            best_centers = np.unravel_index(np.argsort(searchlight.scores_, axis=None)[-self.n_estimators:],
                                            searchlight.scores_.shape)
        else:
            best_centers = np.unravel_index(np.argsort(.1/(-searchlight.scores_ - 1e-30), axis=None)[-self.n_estimators:],
                                            searchlight.scores_.shape)
        self.best_spheres = get_sphere_indices(self.mask_img, np.array(best_centers).T.tolist(), self.radius)

        # for v in range(self.n_estimators):
        #     self.estimators_ += [ESTIMATOR_CATALOG[searchlight.estimator](**self.estimator_params)]
        #     self.estimators_[v].fit(np.array([x.get_data()[self.best_spheres[v]] for x in X]), y)

        estimators = []
        for i in range(self.n_estimators):
            estimator = self._make_estimator(append=False)
            estimators.append(estimator)

        if not isinstance(X[0], nibabel.Nifti1Image):
            X = [nibabel.load(x) for x in X]


        # for v, e in enumerate(estimators):
        #     print(v)
        #     _parallel_build_estimator(e, np.array([x.get_data()[self.best_spheres[v]] for x in X]), y)

        estimators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
                     delayed(_parallel_build_estimator)(e, np.array([x.get_data()[self.best_spheres[v]] for x in X]), y)
                     for v, e in enumerate(estimators))

        self.estimators_ = estimators

        return self

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is a vote by the individual searchlights.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes.
        """

        # votes = []
        # for v in range(self.n_best):
        #     votes += [self.estimators_[v].predict(np.array([x.get_data()[self.best_spheres[v]] for x in X]))]

        if not isinstance(X[0], nibabel.Nifti1Image):
            X = [nibabel.load(x) for x in X]


        votes = Parallel(n_jobs=min(self.n_jobs, len(self.estimators_)), verbose=self.verbose, backend="threading")\
                        (delayed(_vote)(e, np.array([x.get_data()[self.best_spheres[v]] for x in X]), continuous=False)
                         for v, e in enumerate(self.estimators_))


        if self.base_estimator_._estimator_type == 'classifier':
            vote = mode(votes).mode[0]
        else:
            vote = np.mean(votes, axis=0)

        return vote

    def _predict_proba_lr(self, X):
        """Predict class for X.

        The predicted class of an input sample is a vote by the individual searchlights.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes.
        """

        if not isinstance(X[0], nibabel.Nifti1Image):
            X = [nibabel.load(x) for x in X]

        votes = Parallel(n_jobs=min(self.n_jobs, len(self.estimators_)), verbose=self.verbose, backend="threading")\
                        (delayed(_vote)(e, np.array([x.get_data()[self.best_spheres[v]] for x in X]), continuous=True)
                         for v, e in enumerate(self.estimators_))

        vote = np.mean(votes, axis=0)

        return vote

    def decision_function(self, X):
        """Predict class for X.

        The predicted class of an input sample is a vote by the individual searchlights.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes.
        """

        if not isinstance(X[0], nibabel.Nifti1Image):
            X = [nibabel.load(x) for x in X]

        votes = Parallel(n_jobs=min(self.n_jobs, len(self.estimators_)), verbose=self.verbose, backend="threading")\
                        (delayed(_decision_function)(e, np.array([x.get_data()[self.best_spheres[v]] for x in X]))
                         for v, e in enumerate(self.estimators_))

        return np.mean(votes, axis=0)

def _vote(estimator, X, continuous=False):
    """Private function used to compute a single vote in parallel."""

    if estimator._estimator_type == 'classifier':
        if continuous:
            return estimator.decision_function(X)
        else:
            return estimator.predict(X)
    else:
        return estimator.predict(X)

def _decision_function(estimator, X):
    """Private function used to compute a single vote in parallel."""

    return estimator.decision_function(X)

def _predict(estimator, X):
    """Private function used to compute a single vote in parallel."""

    return estimator.predict(X)

def _parallel_build_estimator(estimator, X, y):
    """Private function used to fit a single estimator in parallel."""
    estimator.fit(X, y)

    return estimator


def get_sphere_indices(img_mask, centers, radius):

    """

    :rtype : list of sphere index tuples (nTuples = nCenters)
    :param img_mask: Nifti Image of mask
    :param centers: list of centers [[x1,y1,z1], [x2,y2,z2], ...]
    :param radius: radius in voxels (integer)
    """
    if isinstance(img_mask, str):
        img_mask = nibabel.load(img_mask)

    mask_coords = np.array(list(np.array(np.where(img_mask.get_data() != 0)).T))
    neigh = neighbors.NearestNeighbors(radius=radius)
    neigh.fit(mask_coords)
    ind = neigh.radius_neighbors(centers, return_distance=False)

    return [tuple(mask_coords[i].T) for i in ind]
