import os
import fnmatch
import nibabel
import numpy as np


def get_regressors(dir, labels, file_type=None):

    FILE_TYPES = {'header_files': ['.hdr', 'header', 'hdr'],
                  'nifti_files': ['.nii', 'nifti', 'nii']}

    if file_type is None:
        file_extension = 'nii'
    elif isinstance(file_type, str):
        file_type = file_type.lower()
        for k, v in FILE_TYPES.items():
            if file_type in v:
                file_extension = v[-1]
    else:
        raise ValueError("File type must be specified by string (eg. 'header' or '.nii')!")

    betas = [nibabel.load(os.path.join(dir, filename)) for filename in sorted(os.listdir(dir))
             if fnmatch.fnmatch(filename, 'beta_*.%s' %(file_extension))]
    descrip_last_beta = str(betas[-1].header['descrip'])
    start_ind = descrip_last_beta.find('Sn(')
    end_ind = descrip_last_beta.find(') constant')
    n_runs = int(descrip_last_beta[start_ind + 3:end_ind])

    betas_out, labels_out = [], []
    for i, label in enumerate(labels):
        betas_ = [beta for beta in betas if label in str(beta.header['descrip'])]
        betas_out += betas_
        labels_out += list((i + 1) * np.ones(len(betas_), dtype=np.int8))

    return betas_out, labels_out
