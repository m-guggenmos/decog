from sklearn.neighbors import DistanceMetric
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
import scipy
# import warnings
# warnings.filterwarnings('error')

SKLEARN_METRICS = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean',
                   'mahalanobis']

# METRICS = scipy.spatial.distance.__all__

 # ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’, ‘yule’.

def distance(X, distance_measure='euclidean'):

    X = np.array(X)

    if distance_measure in SKLEARN_METRICS:
        distance_ = DistanceMetric.get_metric(distance_measure).pairwise(X)
    elif distance_measure is 'pearson':
        distance_ = np.corrcoef(X)
    else:
        distance_ = None

    return distance_


def normalization_MEEG(X, baseline, normalization_method='cocktail_blank', axis=0):

    if normalization_method == 'cocktail_blank':
        Xn = X - np.tile(np.mean(X, axis=axis), (X.shape[0], 1))
    elif normalization_method == 'zscore':
        for cond in range(X.shape[0]):
            for trial in range(X.shape[1]):
                for channel in range(X.shape[2]):
                    trial_data = X[cond, trial, channel, :]
                    baseline_mean = np.mean(trial_data[baseline])
                    baseline_std = np.std(trial_data[baseline])
                    # X[cond, trial, channel, :] = (trial_data - baseline_mean) / baseline_std
                    try:
                        X[cond, trial, channel, :] = (trial_data - baseline_mean) / baseline_std
                    except:
                        print('hello')
        Xn = X
    else:
        Xn = None

    return Xn


if __name__ == '__main__':

    X = np.array([[1, 2, 3], [1, 2, 5]])
    result = distance(X, distance_measure='pearson')
    print(result)
