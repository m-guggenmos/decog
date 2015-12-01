from nipy.io.api import load_image,save_image
from nipy.core.image import Image
import numpy as np
import os
from mgni.smooth import smooth
import json

mode = 'SVR'  # SVC or SVR

root = '/data/fmri/simu/'
# roi_name = '%s_Fusiform_posterior_AAL.nii' % mode
roi_name = '%s_wholebrain.nii' % mode
mask = '/home/matteo/science/fmri/roi/example_mask.nii'
roi = '/home/matteo/science/fmri/roi/example_mask.nii'
# roi = '/home/matteo/science/fmri/roi/AAL/rFusiform_posterior_AAL.nii'
n_subjects = 16
fwhm = 10

img_mask = load_image(mask)
img_mask = Image(img_mask.get_data().astype(np.float64), img_mask.coordmap, img_mask.header)
img_roi = load_image(roi)
img_roi = Image(img_roi.get_data().astype(np.float64), img_roi.coordmap, img_roi.header)
ind_roi = img_roi.get_data().astype(bool) & img_mask.get_data().astype(bool)
ind_noroi = ~img_roi.get_data().astype(bool) & img_mask.get_data().astype(bool)

roi_data = np.random.randn(np.sum(ind_roi))

for i in range(1, n_subjects+1):

    print('subject %02g' % i)

    s_dir = os.path.join(root, '%02g' % i)
    if not os.path.isdir(s_dir):
        os.makedirs(s_dir)
    roi_dir = os.path.join(s_dir, 'roi')
    if not os.path.isdir(roi_dir):
        os.makedirs(roi_dir)

    if mode == 'SVR':
        img_mask.get_data()[ind_roi] = (i - (i > n_subjects / 2) * n_subjects / 2) * roi_data + 0.4*np.random.randn(roi_data.shape[0])
        y = (i - (i > n_subjects / 2) * n_subjects / 2)
    else:
        img_mask.get_data()[ind_roi] = (1 + (i > n_subjects / 2)) * roi_data + 0.4\
                                                                               *np.random.randn(roi_data.shape[0])
        y = 1 + (i > n_subjects / 2)
    img_mask.get_data()[ind_noroi] = np.random.randn(np.sum(ind_noroi))
    img_mask = smooth(img_mask, fwhm)

    save_image(img_mask, os.path.join(roi_dir, roi_name))

    json.dump({'y': y}, open(os.path.join(s_dir, 'y_%s.json' % mode), 'w+'))