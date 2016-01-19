import os
import fnmatch
import nibabel
import numpy as np


def get_regressors(dir, labels):

    betas = [nibabel.load(os.path.join(dir, filename)) for filename in sorted(os.listdir(dir)) if fnmatch.fnmatch(filename, 'beta_*.hdr')]
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
