from nilearn.input_data.base_masker import BaseMasker
from nilearn.input_data import NiftiMasker
from nilearn import _utils
from nilearn.image import smooth_img, resample_img
import numpy as np
from nibabel import Nifti1Image, load
import os
import warnings

FDICT = dict(mean=(np.mean, dict(axis=1)), max=(np.max, dict(axis=1)))


class MultiRoiMasker(BaseMasker):
    def __init__(self, mask_img=None, rois=None, smoothing_fwhm=None, resampling=None,
                 normalize=False, combine_rois=False, searchlight=False):

        self.mask_img = mask_img
        if not rois:
            raise Exception('MultiRoiMasker requires at least one ROI!')
        elif not isinstance(rois, (list, tuple)):
            rois = [rois]
        if normalize and searchlight:
            raise ValueError("Combination searchlight=True and normalize=True currently not supported!")
        self.rois = rois
        self.smoothing_fwhm = smoothing_fwhm
        self.resampling = resampling
        self.normalize = normalize
        self.combine_rois = combine_rois
        self.searchlight = searchlight

    def fit(self):

        if self.resampling is not None:
            resample = np.diag(self.resampling * np.ones(3))
        else:
            resample =  None

        self.mask_img = resample_img(self.mask_img, target_affine=resample, interpolation='nearest')

        if not isinstance(self.rois, tuple):
            self.masker = dict()
            for roi_id, roi in enumerate(self.rois):
                if self.resampling is not None:
                    roi = resample_img(roi, target_affine=resample, interpolation='nearest')
                self.masker[roi_id] = NiftiMasker(mask_img=roi)
                self.masker[roi_id].fit()
        else:
            self.masker = [None] * len(self.rois)  # first create as list..
            for m, rois_modality in enumerate(self.rois):
                self.masker[m] = dict()
                for roi_id, roi in enumerate(rois_modality):
                    if self.resampling is not None:
                        roi = resample_img(roi, target_affine=resample, interpolation='nearest')
                    self.masker[m][roi_id] = NiftiMasker(mask_img=roi)
                    self.masker[m][roi_id].fit()
            self.masker = tuple(self.masker)  # .. then make conform again

        return self

    def transform(self, X, confounds=None):
        """

        Parameters
        ----------
        X: list of Niimg-like objects
        """

        X = self.preprocess(X)

        if self.searchlight:
            affine = self.mask.affine
            shape = self.mask.shape

        if self.combine_rois:
            if not self.searchlight:
                data = {0: np.concatenate([masker.transform(X) for masker in self.masker.values()], axis=1)}
                if self.normalize:
                    data = {0: np.array([d - np.mean(d) for d in data[0]])}
            else:
                data = {0: ([self.masker[0].transform(x) for x in X],
                            Nifti1Image(np.sum([load(roi).get_data() for roi in self.rois], axis=0), affine=affine))}
        else:
            if not self.searchlight:
                if not isinstance(self.masker, tuple):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        data = {k: masker.transform(X) for k, masker in self.masker.items()}
                        if self.normalize:
                            data = {k: np.array([d-np.mean(d) for d in v]) for k, v in data.items()}
                else:
                    data = dict()
                    c = 0
                    for m, modality in enumerate(self.masker):
                        X_m = [x[m] for x in X]
                        for k, masker in modality.items():
                            data_ = masker.transform(X_m)
                            if self.normalize:
                                data_ = np.array([d - np.mean(d) for d in data_])
                            data.update({c: data_})
                            c += 1
            else:
                data = {k: (masker.transform(X), self.rois[k]) for k, masker in self.masker.items()}
        return data

    def preprocess(self, imgs):

        smooth_prefix = '' if self.smoothing_fwhm is None else 's%g' % self.smoothing_fwhm
        resample_prefix = '' if self.resampling is None else 'r%g' % self.resampling

        if not isinstance(imgs, list):
            imgs = [imgs]

        path_first = imgs[0] if isinstance(imgs[0], str) else imgs[0].get_filename()

        path_first_resampled = os.path.join(os.path.dirname(path_first), resample_prefix + os.path.basename(path_first))
        path_first_smoothed = os.path.join(os.path.dirname(path_first), smooth_prefix + resample_prefix + os.path.basename(path_first))

        if self.resampling is not None or self.smoothing_fwhm is not None:
            if self.resampling is not None and not os.path.exists(path_first_smoothed):
                if not os.path.exists(path_first_resampled):
                    imgs = resample_img(imgs, target_affine=np.diag(self.resampling * np.ones(3)))
                else:
                    imgs = [os.path.join(os.path.dirname(img), resample_prefix + os.path.basename(img)) if isinstance(img, str)
                            else os.path.join(os.path.dirname(img.get_filename()), resample_prefix + os.path.basename(img.get_filename())) for img in imgs]
            if self.smoothing_fwhm is not None:
                if not os.path.exists(path_first_smoothed):
                    imgs = smooth_img(imgs, self.smoothing_fwhm)
                else:
                    imgs = [os.path.join(os.path.dirname(img), smooth_prefix + resample_prefix + os.path.basename(img)) if isinstance(img, str)
                            else os.path.join(os.path.dirname(img.get_filename()), smooth_prefix + resample_prefix + os.path.basename(img.get_filename())) for img in imgs]
        else:
            imgs = [check_niimg_3d(img) for img in imgs]

        return imgs




class MultiRoiExtractMasker(BaseMasker):
    def __init__(self, mask_img=None, extract_funcs=None, smoothing_fwhm=None):

        if mask_img is None:
            raise Exception('MultiRoiMasker requires at least one ROI!')

        self.extract_funcs = extract_funcs
        self.mask_img = mask_img
        self.smoothing_fwhm = smoothing_fwhm

    def fit(self):

        self.mask_img_ = [_utils.check_niimg_3d(img) for img in self.mask_img]

        return self

    def transform(self, imgs, confounds=None):
        """

        Parameters
        ----------
        imgs: list of Niimg-like objects
        """
        self._check_fitted()

        if self.smoothing_fwhm:
            imgs = smooth_img(imgs, self.smoothing_fwhm)

        imgs = [_utils.check_niimg_3d(img) for img in imgs]

        for i, roi in enumerate(self.mask_img_):
            masker = NiftiMasker(mask_img=roi)
            x = masker.fit_transform(imgs)
            if self.extract_funcs is not None:
                x = np.array([FDICT[f][0](x, **FDICT[f][1]) for f in self.extract_funcs])
            if i == 0:
                X = x
            else:
                X = np.concatenate((X, x), axis=0)

        return X.swapaxes(0, 1)



from nilearn.input_data.base_masker import BaseMasker
from nilearn.image import smooth_img, resample_img
from nilearn._utils import check_niimg_3d

class DummyMasker(BaseMasker):

    def __init__(self):

        self.mask_img = None

    def fit(self, X, y):

        """
        Dummy fit function
        """
        self.mask_img_ = None

        return self

    def transform(self, X, confounds=None):
        """
        Dummy transform function
        """

        return X


class SearchlightMasker(BaseMasker):

    def __init__(self, smoothing_fwhm=None, target_affine=None):

        self.smoothing_fwhm = smoothing_fwhm
        self.target_affine = target_affine

    def fit(self):

        return self

    def transform(self, imgs, confounds=None):
        if self.smoothing_fwhm is not None or self.target_affine is not None:
            if self.smoothing_fwhm is not None:
                imgs = smooth_img(imgs, self.smoothing_fwhm)
            if self.target_affine is not None:
                imgs = resample_img(imgs, target_affine=self.target_affine)
        else:
            imgs = [check_niimg_3d(img) for img in imgs] if isinstance(imgs, list) else check_niimg_3d(imgs)

        return imgs


class SmoothResampleMasker(BaseMasker):

    def __init__(self, mask_img=None, smoothing_fwhm=None, resampling=None, searchlight=False):

        self.mask_img = mask_img
        self.smoothing_fwhm = smoothing_fwhm
        self.resampling = resampling
        self.searchlight = searchlight

        self.masker = None

    def fit(self):

        if self.resampling is not None:
            self.mask_img = resample_img(self.mask_img, target_affine=np.diag(self.resampling * np.ones(3)))
        self.masker = NiftiMasker(mask_img=self.mask_img)
        self.masker.fit()

        return self

    def transform(self, imgs, confounds=None):

        smooth_prefix = '' if self.smoothing_fwhm is None else 's%g' % self.smoothing_fwhm
        resample_prefix = '' if self.smoothing_fwhm is None else 'r%g' % self.smoothing_fwhm

        if not isinstance(imgs, list):
            imgs = [imgs]

        path_first = imgs[0] if isinstance(imgs[0], str) else imgs[0].get_filename()

        path_first_resampled = os.path.join(os.path.dirname(path_first), resample_prefix + os.path.basename(path_first))
        path_first_smoothed = os.path.join(os.path.dirname(path_first), smooth_prefix + resample_prefix + os.path.basename(path_first))

        if self.resampling is not None and self.smoothing_fwhm is not None:
            if self.resampling is not None:
                if not os.path.exists(path_first_resampled) and not os.path.exists(path_first_smoothed):
                    imgs = resample_img(imgs, target_affine=np.diag(self.resampling * np.ones(3)))
                else:
                    imgs = []
            if self.smoothing_fwhm is not None:
                if not os.path.exists(path_first_smoothed):
                    imgs = smooth_img(imgs, self.smoothing_fwhm)
                else:
                    imgs = []
        else:
            imgs = [check_niimg_3d(img) for img in imgs]

        return self.masker.transform(imgs)
