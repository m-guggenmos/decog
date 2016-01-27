from nilearn.input_data.base_masker import BaseMasker
from nilearn.input_data import NiftiMasker
from nilearn import _utils
from nilearn.image import smooth_img, resample_img
import numpy as np
from nibabel import Nifti1Image, load
from collections import OrderedDict, Sequence

FDICT = dict(mean=(np.mean, dict(axis=1)), max=(np.max, dict(axis=1)))


class MultiRoiMasker(BaseMasker):
    def __init__(self, mask_img=None, rois=None, smoothing_fwhm=None, resample=None, combine_rois=False,
                 searchlight=False):

        self.mask_img = mask_img
        if not rois:
            raise Exception('MultiRoiMasker requires at least one ROI!')
        elif not isinstance(rois, Sequence):
            rois = [rois]
        self.rois = rois
        self.smoothing_fwhm = smoothing_fwhm
        self.resample = resample
        self.combine_rois = combine_rois
        self.searchlight = searchlight

    def fit(self):

        if self.resample is not None:
            self.desired_affine = np.diag(self.resample * np.ones(3))
        else:
            self.desired_affine = None

        if self.searchlight:
            self.mask = load(self.mask_img)
            masker = SearchlightMasker(smoothing_fwhm=self.smoothing_fwhm, target_affine=self.desired_affine)

        if not isinstance(self.rois, tuple):
            self.masker = dict()
            for roi_id, roi in enumerate(self.rois):
                if self.searchlight:
                    self.masker[roi_id] = masker
                else:
                    self.masker[roi_id] = NiftiMasker(mask_img=resample_img(roi, interpolation='nearest', target_affine=self.desired_affine),
                                                      smoothing_fwhm=self.smoothing_fwhm)
                self.masker[roi_id].fit()
        else:
            self.masker = [None] * len(self.rois)  # first create as list..
            for m, rois_modality in enumerate(self.rois):
                self.masker[m] = dict()
                for roi_id, roi in enumerate(rois_modality):
                    self.masker[m][roi_id] = NiftiMasker(mask_img=resample_img(roi, interpolation='nearest', target_affine=self.desired_affine),
                                                         smoothing_fwhm=self.smoothing_fwhm)
                    self.masker[m][roi_id].fit()
            self.masker = tuple(self.masker)  # .. then make conform again

        return self

    def transform(self, X, confounds=None):
        """

        Parameters
        ----------
        X: list of Niimg-like objects
        """

        if self.resample is not None:
            X = resample_img(X, target_affine=self.desired_affine)

        if self.searchlight:
            affine = self.mask.affine
            shape = self.mask.shape

        if self.combine_rois:
            if self.searchlight:
#                 ValueError("""Illegal combination of searchlight=True and combine_rois=True: if searchlight should be
# applied to multiple combined ROIs, combine the ROIs first and supply them as a single processing mask.
# MultiRoiMasker is the wrong place in this case!""")
                return {0: ([self.masker[0].transform(x) for x in X],
                            Nifti1Image(np.sum([load(roi).get_data() for roi in self.rois], axis=0), affine=affine))}
                    # [Nifti1Image(np.sum([np.reshape(masker.transform(x), shape) for masker in self.masker.values()], axis=0), affine=affine)
                    #     for x in X]
            else:
                return {0: np.concatenate([masker.transform(X) for masker in self.masker.values()], axis=1)}
        else:
            if self.searchlight:
                return {k: (masker.transform(X), self.rois[k]) for k, masker in self.masker.items()}
            else:
                if not isinstance(self.masker, tuple):
                    return {k: masker.transform(X) for k, masker in self.masker.items()}
                else:
                    data = dict()
                    c = 0
                    for m, modality in enumerate(self.masker):
                        X_m = [x[m] for x in X]
                        for k, masker in modality.items():
                            data.update({c: masker.transform(X_m)})
                            c += 1
                    return data



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
