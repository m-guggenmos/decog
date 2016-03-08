from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn import neighbors
from sklearn.metrics import make_scorer
from nilearn.decoding.searchlight import search_light
from nilearn import masking
import nibabel
import numpy as np
from scipy.stats import pearsonr
from nilearn.input_data import NiftiMasker
from warnings import warn
from functools import partial
import warnings

ESTIMATOR_CATALOG = dict(svc=svm.LinearSVC, svr=svm.SVR)


class SearchLight(BaseEstimator):
    """ Note: this is an almost-clone of nilearn's searchlight algorithm, so all credit goes to
     them!

    Implement search_light analysis using an arbitrary type of classifier.

    Parameters
    -----------
    mask_img : Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        boolean image giving location of voxels containing usable signals.

    process_mask_img : Niimg-like object, optional
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        boolean image giving voxels on which searchlight should be
        computed.

    radius : float, optional
        radius of the searchlight ball, in millimeters. Defaults to 2.

    estimator : 'svr', 'svc', or an estimator object implementing 'fit'
        The object to use to fit the data

    n_jobs : int, optional. Default is -1.
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    scoring : string or callable, optional
        The scoring strategy to use. See the scikit-learn documentation
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied.

    verbose : int, optional
        Verbosity level. Defaut is False

    Notes
    ------
    The searchlight [Kriegeskorte 06] is a widely used approach for the
    study of the fine-grained patterns of information in fMRI analysis.
    Its principle is relatively simple: a small group of neighboring
    features is extracted from the data, and the prediction function is
    instantiated on these features only. The resulting prediction
    accuracy is thus associated with all the features within the group,
    or only with the feature on the center. This yields a map of local
    fine-grained information, that can be used for assessing hypothesis
    on the local spatial layout of the neural code under investigation.

    Nikolaus Kriegeskorte, Rainer Goebel & Peter Bandettini.
    Information-based functional brain mapping.
    Proceedings of the National Academy of Sciences
    of the United States of America,
    vol. 103, no. 10, pages 3863-3868, March 2006
    """

    def __init__(self, mask_img, process_mask_img=None, radius=2.,
                 estimator='svc', estimator_params=None,
                 n_jobs=-1, scoring=None, cv=None,
                 verbose=0):
        self.mask_img = mask_img
        self.process_mask_img = process_mask_img
        self.radius = radius
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        self.affine = None

    @property
    def estimator_params(self):
        return self._estimator_params

    @estimator_params.setter
    def estimator_params(self, value):
        if value is None:
            self._estimator_params = dict()
        else:
            self._estimator_params = value

    def fit(self, imgs, y):
        """Fit the searchlight

        Parameters
        ----------
        imgs : List of images
        y : 1D array-like
            Target variable to predict. Must have exactly as many elements as
            3D images in img.

        Attributes
        ----------
        `scores_` : numpy.ndarray
            search_light scores. Same shape as input parameter
            process_mask_img.
        """

        # Compute world coordinates of all in-mask voxels.
        self.affine = nibabel.load(imgs[0]).affine if isinstance(imgs[0], str) else imgs[0].affine
        mask, mask_affine = masking._load_mask_img(self.mask_img)
        mask_coords = np.where(mask != 0)
        mask_coords = np.asarray(mask_coords + (np.ones(len(mask_coords[0]),
                                                        dtype=np.int),))
        mask_coords = np.dot(mask_affine, mask_coords)[:3].T
        # Compute world coordinates of all in-process mask voxels
        if self.process_mask_img is None:
            process_mask = mask
            process_mask_coords_world = mask_coords
        else:
            process_mask, process_mask_affine = \
                masking._load_mask_img(self.process_mask_img)
            process_mask_coords = np.where(process_mask != 0)
            process_mask_coords_world = \
                np.asarray(process_mask_coords
                           + (np.ones(len(process_mask_coords[0]),
                                      dtype=np.int),))
            process_mask_coords_world = np.dot(process_mask_affine,
                                         process_mask_coords_world)[:3].T

        clf = neighbors.NearestNeighbors(radius=self.radius)
        A = clf.fit(mask_coords).radius_neighbors_graph(process_mask_coords_world)
        if self.process_mask_img is not None:
            empty_ind = [i for i in range(A.shape[0]) if A[i,:].getnnz() == 0]
            if empty_ind:
                warn('Skipping %g voxels of processing mask outside mask_img (first index: %g)' % (len(empty_ind), empty_ind[0]))
                A = A[list(set(range(A.shape[0])) - set(empty_ind)), :]
                process_mask[tuple(np.asarray(process_mask_coords)[:, empty_ind])] = False

        A = A.tolil()

        # scores is an 1D array of CV scores with length equals to the number
        # of voxels in processing mask (columns in process_mask)
        # X = masking._apply_mask_fmri(imgs, self.mask_img)
        X = NiftiMasker(mask_img=self.mask_img).fit_transform(imgs)

        estimator = self.estimator
        if isinstance(estimator, str):
            estimator = ESTIMATOR_CATALOG[estimator](**self.estimator_params)
        elif hasattr(estimator, '__name__'): # check if estimator has to be initialized
            estimator = estimator(**self.estimator_params)
        else:
            estimator.set_params(**self.estimator_params)

        scores = search_light(X, y, estimator, A,
                              self.scoring, self.cv, self.n_jobs,
                              self.verbose)
        scores_3D = np.zeros(process_mask.shape)
        scores_3D[process_mask] = scores
        self.scores_ = scores_3D
        return self

    def predict(self, X):
        return nibabel.Nifti1Image(self.scores_, affine=self.affine)


# Note: the peculiar format of creating these scorers is necessary in order to be compatible with
# multiprocessing_on_dill

def _accuracy_minus_chance_func(y, y_pred, chance_level=0.5):
    return np.mean(np.array(y) == np.array(y_pred)) - chance_level


def accuracy_minus_chance_scorer(chance_level=0.5):
    func = partial(_accuracy_minus_chance_func, chance_level=chance_level)
    func.__name__ = 'accuracy_minus_chance'
    return make_scorer(func)


def _pearson_func(y, y_pred):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = pearsonr(y, y_pred)[0]
    return result


def pearson_scorer():
    return make_scorer(_pearson_func)

