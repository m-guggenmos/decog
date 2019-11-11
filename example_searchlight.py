import os
import numpy as np
import nibabel
from decereb.chain import SimpleChain, Data
from decereb.estimators.searchlight import SearchLight
from nilearn import datasets
from nilearn.image import index_img
from nilearn._utils import check_niimg_4d
from sklearn.cross_validation import KFold

haxby_dataset = datasets.fetch_haxby_simple()

conditions = np.recfromtxt(haxby_dataset.conditions_target)['f0']
condition_mask = np.logical_or(conditions == beta'face', conditions == beta'house')
labels = conditions[condition_mask]

fmri_img = nibabel.load(haxby_dataset.func)
fmri_img = index_img(fmri_img, condition_mask)
fmri_img = [img for img in check_niimg_4d(fmri_img, return_iterator=True)]

mask_img = nibabel.load(haxby_dataset.mask)

cv = KFold(len(labels), n_folds=4)
clf_args = dict(mask_img=mask_img, process_mask_img=mask_img, cv=cv, radius=5.6)
data = Data(data=fmri_img, labels=labels)
chain = SimpleChain(clf=SearchLight, clf_args=clf_args, data=data)

root = os.path.dirname(haxby_dataset['session_target'])
output_path = os.path.join(root, 'searchlight.nii')

chain.run(n_jobs_folds=1, verbose=3, output_path=output_path)