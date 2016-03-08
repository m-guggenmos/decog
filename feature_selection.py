from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import safe_mask, check_array
from sklearn.utils.sparsefuncs import mean_variance_axis
from warnings import warn
import numpy as np
from decereb.estimators import RoiEnsemble

class MultiRoiVarianceThreshold(BaseEstimator, SelectorMixin):
    """Feature selector that removes all low-variance features.

    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.

    Read more in the :ref:`User Guide <variance_threshold>`.

    Parameters
    ----------
    threshold : float, optional
        Features with a training-set variance lower than this threshold will
        be removed. The default is to keep all features with non-zero variance,
        i.e. remove the features that have the same value in all samples.

    Attributes
    ----------
    variances_ : array, shape (n_features,)
        Variances of individual features.
    """

    def __init__(self, threshold=0., allow_empty_roi=True):
        self.threshold = threshold
        self.allow_empty_roi = allow_empty_roi

    def fit(self, X, y=None):
        """Learn empirical variances from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Sample vectors from which to compute variances.

        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self
        """
        if not isinstance(X, dict):
            raise ValueError('X must be a dict')

        self.variances_ = {k: np.var(x, axis=0) for k, x in X.items()}

        roi_no_sample = [k for k, v in self.variances_.items() if np.all(v <= self.threshold)]
        self.roi_id_valid = np.setdiff1d(list(self.variances_.keys()), roi_no_sample, assume_unique=True)
        if len(roi_no_sample) == len(self.variances_):
            raise ValueError("No feature in any roi meets the variance threshold %.5f" % self.threshold)
        elif np.any(roi_no_sample):
            message = "No feature in roi(s) %s meets the variance threshold %.5f" % (list(roi_no_sample), self.threshold)
            if self.allow_empty_roi:
                warn(message, UserWarning)
            else:
                raise ValueError(message)

        # for i, var in enumerate(self.variances_):
        #     if np.all(var <= self.threshold):
        #         msg = "No feature in roi %g meets the variance threshold {0:.5f}" % i
        #         if X[i].shape[0] == 1:
        #             msg += " (roi contains only one sample)"
        #         raise ValueError(msg.format(self.threshold))

        return self

    def transform(self, X):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
        """
        masks = self.get_support()
        X_r = dict()
        for roi_id in self.roi_id_valid:
            mask = masks[roi_id]
            if len(mask) != X[roi_id].shape[1]:
                raise ValueError("Roi %g has a different shape than during fitting." % roi_id)
            if not mask.any():
                warn("No features were selected in roi %g: either the data is"
                     " too noisy or the selection test too strict." % roi_id,
                     UserWarning)
                X_r[roi_id] = np.empty(0).reshape((X[roi_id].shape[0], 0))
            else:
                X_r[roi_id] = X[roi_id][:, safe_mask(X[roi_id], mask)]
        return X_r

    def _get_support_mask(self):

        support_masks = {k: v > self.threshold for k, v in self.variances_.items()}

        return support_masks


class MultiRoiVariancePercentile(BaseEstimator, SelectorMixin):
    """Feature selector that removes all low-variance features.

    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.

    Read more in the :ref:`User Guide <variance_threshold>`.

    Parameters
    ----------
    threshold : float, optional
        Features with a training-set variance lower than this threshold will
        be removed. The default is to keep all features with non-zero variance,
        i.e. remove the features that have the same value in all samples.

    Attributes
    ----------
    variances_ : array, shape (n_features,)
        Variances of individual features.
    """

    def __init__(self, threshold=0., allow_empty_roi=True):
        self.threshold = threshold
        self.allow_empty_roi = allow_empty_roi

    def fit(self, X, y=None):
        """Learn empirical variances from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Sample vectors from which to compute variances.

        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self
        """
        if not isinstance(X, dict):
            raise ValueError('X must be a dict')

        self.variances_ = {k: np.var(x, axis=0) for k, x in X.items()}

        # for i, var in enumerate(self.variances_):
        #     if np.all(var <= self.threshold):
        #         msg = "No feature in roi %g meets the variance threshold {0:.5f}" % i
        #         if X[i].shape[0] == 1:
        #             msg += " (roi contains only one sample)"
        #         raise ValueError(msg.format(self.threshold))

        return self

    def transform(self, X):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
        """
        masks = self.get_support()
        X_r = dict()
        for roi_id in self.variances_.keys():
            mask = masks[roi_id]
            if len(mask) != X[roi_id].shape[1]:
                raise ValueError("Roi %g has a different shape than during fitting." % roi_id)
            X_r[roi_id] = X[roi_id][:, safe_mask(X[roi_id], mask)]
        return X_r

    def _get_support_mask(self):

        support_masks = {k: v > np.percentile(v, self.threshold) for k, v in self.variances_.items()}

        return support_masks


from sklearn.feature_selection.univariate_selection import _BaseFilter, f_classif, _clean_nans
from scipy import stats

class MuliRoiSelectPercentile(_BaseFilter):
    """Select features according to a percentile of the highest scores.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    score_func : callable
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues).

    percentile : int, optional, default=10
        Percent of features to keep.

    Attributes
    ----------
    scores_ : array-like, shape=(n_features,)
        Scores of features.

    pvalues_ : array-like, shape=(n_features,)
        p-values of feature scores.

    Notes
    -----
    Ties between features with equal scores will be broken in an unspecified
    way.

    See also
    --------
    f_classif: ANOVA F-value between labe/feature for classification tasks.
    chi2: Chi-squared stats of non-negative features for classification tasks.
    f_regression: F-value between label/feature for regression tasks.
    SelectKBest: Select features based on the k highest scores.
    SelectFpr: Select features based on a false positive rate test.
    SelectFdr: Select features based on an estimated false discovery rate.
    SelectFwe: Select features based on family-wise error rate.
    GenericUnivariateSelect: Univariate feature selector with configurable mode.
    """

    def __init__(self, score_func=f_classif, percentile=10):
        super(MuliRoiSelectPercentile, self).__init__(score_func)
        self.percentile = percentile

    def fit(self, X, y):
        """Run score function on (X, y) and get the appropriate features.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        if not callable(self.score_func):
            raise TypeError("The score function should be a callable, %s (%s) "
                            "was passed."
                            % (self.score_func, type(self.score_func)))

        self.scores_ = dict()
        self.pvalues_ = dict()

        for k, x in X.items():
            score_, pvalue_ = self.score_func(x, y)
            self.scores_[k] = np.asarray(score_)
            self.pvalues_[k] = np.asarray(pvalue_)

        return self

    def transform(self, X):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
        """
        masks = self.get_support()
        X_r = dict()
        for roi_id, mask in masks.items():
            if len(mask) != X[roi_id].shape[1]:
                raise ValueError("Roi %g has a different shape than during fitting." % roi_id)
            if not mask.any():
                warn("No features were selected in roi %g: either the data is"
                     " too noisy or the selection test too strict." % roi_id,
                     UserWarning)
                X_r[roi_id] = np.empty(0).reshape((X[roi_id].shape[0], 0))
            else:
                X_r[roi_id] = X[roi_id][:, safe_mask(X[roi_id], mask)]
        return X_r

    def _check_params(self, X, y):
        if not 0 <= self.percentile <= 100:
            raise ValueError("percentile should be >=0, <=100; got %r"
                             % self.percentile)

    def _get_support_mask(self):
        check_is_fitted(self, 'scores_')

        # Cater for NaNs
        if self.percentile == 100:
            return {k: np.ones(len(score), dtype=np.bool) for k, score in self.scores_.items()}
        elif self.percentile == 0:
            return {k: np.zeros(len(score), dtype=np.bool) for k, score in self.scores_.items()}

        masks = dict()
        for roi_id, score in self.scores_.items():
            scores = _clean_nans(score)
            if len(scores) == 1:
                mask = np.array([True])
            else:
                treshold = stats.scoreatpercentile(scores,
                                                   100 - self.percentile)
                mask = scores > treshold
                ties = np.where(scores == treshold)[0]
                if len(ties):
                    max_feats = len(scores) * self.percentile // 100
                    kept_ties = ties[:max_feats - mask.sum()]
                    mask[kept_ties] = True
            masks[roi_id] = mask
        return masks

from sklearn.utils.validation import NotFittedError, check_is_fitted
from sklearn.feature_selection.from_model import _calculate_threshold, _get_feature_importances
from sklearn.base import clone

class MultiRoiSelectFromModel(BaseEstimator, SelectorMixin):
    """Meta-transformer for selecting features based on importance weights.

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
        This can be both a fitted (if ``prefit`` is set to True)
        or a non-fitted estimator.

    threshold : string, float, optional default None
        The threshold value to use for feature selection. Features whose
        importance is greater or equal are kept while the others are
        discarded. If "median" (resp. "mean"), then the ``threshold`` value is
        the median (resp. the mean) of the feature importances. A scaling
        factor (e.g., "1.25*mean") may also be used. If None and if the
        estimator has a parameter penalty set to l1, either explicitly
        or implicity (e.g, Lasso), the threshold is used is 1e-5.
        Otherwise, "mean" is used by default.

    prefit : bool, default False
        Whether a prefit model is expected to be passed into the constructor
        directly or not. If True, ``transform`` must be called directly
        and SelectFromModel cannot be used with ``cross_val_score``,
        ``GridSearchCV`` and similar utilities that clone the estimator.
        Otherwise train the model using ``fit`` and then ``transform`` to do
        feature selection.

    Attributes
    ----------
    `estimator_`: an estimator
        The base estimator from which the transformer is built.
        This is stored only when a non-fitted estimator is passed to the
        ``SelectFromModel``, i.e when prefit is False.

    `threshold_`: float
        The threshold value used for feature selection.
    """
    def __init__(self, estimator, threshold=None, allow_empty_roi=True):
        self.estimator = estimator
        self.threshold = threshold
        self.allow_empty_roi = allow_empty_roi

    def fit(self, X, y=None, **fit_params):
        """Fit the SelectFromModel meta-transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        **fit_params : Other estimator specific parameters

        Returns
        -------
        self : object
            Returns self.
        """

        if not isinstance(X, dict):
            raise ValueError('X must be a dict')

        if isinstance(self.estimator, RoiEnsemble):
            self.estimator_ = self.estimator
            self.estimator_.fit(X, y, **fit_params)
        else:
            self.estimator_ = dict()
            for roi_id, x in X.items():
                self.estimator_[roi_id] = clone(self.estimator)
                self.estimator_[roi_id].fit(x, y, **fit_params)

        self.masks = self.get_support()

        roi_no_sample = [k for k, mask in self.masks.items() if not np.any(mask)]
        self.roi_id_valid = np.setdiff1d(list(self.masks.keys()), roi_no_sample, assume_unique=True)
        if len(roi_no_sample) == len(self.masks):
            raise ValueError("No feature in any roi meets the required threshold")
        elif np.any(roi_no_sample):
            message = "No feature in roi(s) %s meets the threshold(s) %s" % \
                      (roi_no_sample, [self.threshold_[k] for k in roi_no_sample])
            if self.allow_empty_roi:
                warn(message, UserWarning)
            else:
                raise ValueError(message)


        return self

    def partial_fit(self, X, y=None, **fit_params):
        """Fit the SelectFromModel meta-transformer only once.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        **fit_params : Other estimator specific parameters

        Returns
        -------
        self : object
            Returns self.
        """

        raise RuntimeError('Partial Fit not supported')

        # if not isinstance(X, list):
        #     raise ValueError('X must be a list')
        #
        # self.estimator_ = dict()
        # for roi_id, x in X.items():
        #     self.estimator_[roi_id] = clone(self.estimator)
        #     self.estimator_[roi_id].partial_fit(x, y, **fit_params)
        #
        # return self

    def transform(self, X):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
        """
        X_r = dict()
        for roi_id in self.roi_id_valid:
            if len(self.masks[roi_id]) != X[roi_id].shape[1]:
                raise ValueError("Roi %g has a different shape than during fitting." % roi_id)
            if not self.masks[roi_id].any():
                warn("No features were selected in roi %g: either the data is"
                     " too noisy or the selection test too strict." % roi_id,
                     UserWarning)
                X_r[roi_id] = np.empty(0).reshape((X[roi_id].shape[0], 0))
            else:
                X_r[roi_id] = X[roi_id][:, safe_mask(X[roi_id], self.masks[roi_id])]
        return X_r



    def _get_support_mask(self):
        if hasattr(self, 'estimator_'):
            if isinstance(self.estimator_, dict):
                estimators = self.estimator_
            else:
                estimators = self.estimator_.estimators_
        else:
            raise NotFittedError('Fit the model before transform')

        self.threshold_, masks = dict(), dict()
        for roi_id, estimator in estimators.items():
            score_ = _get_feature_importances(estimator)
            self.threshold_[roi_id] = _calculate_threshold(estimator, score_, self.threshold)
            masks[roi_id] = np.atleast_1d(score_ >= self.threshold_[roi_id])
        return masks


class SelectRoisFromModel(BaseEstimator, SelectorMixin):
    """Meta-transformer for selecting features based on importance weights.

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
        This can be both a fitted (if ``prefit`` is set to True)
        or a non-fitted estimator.

    threshold : string, float, optional default None
        The threshold value to use for feature selection. Features whose
        importance is greater or equal are kept while the others are
        discarded. If "median" (resp. "mean"), then the ``threshold`` value is
        the median (resp. the mean) of the feature importances. A scaling
        factor (e.g., "1.25*mean") may also be used. If None and if the
        estimator has a parameter penalty set to l1, either explicitly
        or implicity (e.g, Lasso), the threshold is used is 1e-5.
        Otherwise, "mean" is used by default.

    prefit : bool, default False
        Whether a prefit model is expected to be passed into the constructor
        directly or not. If True, ``transform`` must be called directly
        and SelectFromModel cannot be used with ``cross_val_score``,
        ``GridSearchCV`` and similar utilities that clone the estimator.
        Otherwise train the model using ``fit`` and then ``transform`` to do
        feature selection.

    Attributes
    ----------
    `estimator_`: an estimator
        The base estimator from which the transformer is built.
        This is stored only when a non-fitted estimator is passed to the
        ``SelectFromModel``, i.e when prefit is False.

    `threshold_`: float
        The threshold value used for feature selection.
    """
    def __init__(self, estimator, criterion=None):
        self.estimator = estimator
        self.criterion = criterion

    def fit(self, X, y=None, **fit_params):
        """Fit the SelectFromModel meta-transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        **fit_params : Other estimator specific parameters

        Returns
        -------
        self : object
            Returns self.
        """

        if not isinstance(X, dict):
            raise ValueError('X must be a dict')

        if isinstance(self.estimator, RoiEnsemble):
            self.estimator_ = self.estimator
            self.estimator_.fit(X, y, **fit_params)
        else:
            self.estimator_ = dict()
            for roi_id, x in X.items():
                self.estimator_[roi_id] = clone(self.estimator)
                self.estimator_[roi_id].fit(x, y, **fit_params)

        self.masks = self.get_support()

        return self

    def partial_fit(self, X, y=None, **fit_params):
        """Fit the SelectFromModel meta-transformer only once.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        **fit_params : Other estimator specific parameters

        Returns
        -------
        self : object
            Returns self.
        """

        raise RuntimeError('Partial Fit not supported')

        # if not isinstance(X, list):
        #     raise ValueError('X must be a list')
        #
        # self.estimator_ = dict()
        # for roi_id, x in X.items():
        #     self.estimator_[roi_id] = clone(self.estimator)
        #     self.estimator_[roi_id].partial_fit(x, y, **fit_params)
        #
        # return self

    def transform(self, X):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
        """

        return {k: X[k] for k in self.masks}



    def _get_support_mask(self):
        if hasattr(self, 'estimator_'):
            if isinstance(self.estimator_, dict):
                estimators = self.estimator_
            else:
                estimators = self.estimator_.estimators_
        else:
            raise NotFittedError('Fit the model before transform')

        # if len(estimators) is already 1, no further feature selection reasonable
        if self.criterion is None or len(estimators) == 1:
            if len(estimators) == 1:
                warn('Skipping ROI feature selection, because otherwise no ROI would be left.')
            return list(estimators.keys())
        else:
            scores = dict()
            for roi_id, estimator in estimators.items():
                scores[roi_id] = np.mean(_get_feature_importances(estimator))

            scores_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            if self.criterion < 1:  # proportion
                return [x[0] for x in scores_sorted[:max(1, round(self.criterion * len(scores)))]]
            else:
                return [x[0] for x in scores_sorted[:self.criterion]]

    # def _get_support_mask(self):
    #     if not hasattr(self, 'estimator_'):
    #         raise NotFittedError('Fit the model before transform')
    #
    #     if isinstance(self.estimator_, dict):
    #         if self.criterion is None:
    #             return list(self.estimator_.keys())
    #         else:
    #             scores = dict()
    #             for roi_id, estimator in self.estimator_.items():
    #                 scores[roi_id] = np.mean(_get_feature_importances(estimator))
    #
    #             scores_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    #
    #             if self.criterion < 1:  # proportion
    #                 return [x[0] for x in scores_sorted[:max(1, round(self.criterion * len(scores)))]]
    #             else:
    #                 return [x[0] for x in scores_sorted[:self.criterion]]
    #     else:
    #         if self.criterion is None:
    #             return list(self.estimator_.keys())
    #         else:




class VariancePercentile(BaseEstimator, SelectorMixin):
    """Feature selector that removes all low-variance features.

    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.

    Read more in the :ref:`User Guide <variance_threshold>`.

    Parameters
    ----------
    threshold : float, optional
        Features with a training-set variance lower than this threshold will
        be removed. The default is to keep all features with non-zero variance,
        i.e. remove the features that have the same value in all samples.

    Attributes
    ----------
    variances_ : array, shape (n_features,)
        Variances of individual features.

    Examples
    --------
    The following dataset has integer features, two of which are the same
    in every sample. These are removed with the default setting for threshold::

        >>> X = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
        >>> selector = VarianceThreshold()
        >>> selector.fit_transform(X)
        array([[2, 0],
               [1, 4],
               [1, 1]])
    """

    def __init__(self, threshold=0.):
        self.threshold = threshold

    def fit(self, X, y=None):
        """Learn empirical variances from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Sample vectors from which to compute variances.

        y : any
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self
        """
        X = check_array(X, ('csr', 'csc'), dtype=np.float64)

        if hasattr(X, "toarray"):   # sparse matrix
            _, self.variances_ = mean_variance_axis(X, axis=0)
        else:
            self.variances_ = np.var(X, axis=0)

        return self

    def _get_support_mask(self):
        check_is_fitted(self, 'variances_')
        support_mask = self.variances_ > np.percentile(self.variances_, self.threshold)
        return support_mask
