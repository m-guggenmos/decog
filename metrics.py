import numpy as np
from scipy.optimize import bisect
from scipy.stats import beta
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics.classification import _check_targets, recall_score
import itertools

def balanced_accuracy_score(y_true, y_pred, sample_weight=None):
    """Compute the balanced pred
    The balanced pred is used in binary classification problems to deal
    with imbalanced datasets. It is defined as the arithmetic mean of sensitivity
    (true positive rate) and specificity (true negative rate), or the average_flat
    pred obtained on either class.
    The best value is 1 and the worst value is 0.
    Read more in the :ref:`User Guide <balanced_accuracy_score>`.
    Parameterspartial(power, exponent=2)
    ----------
    y_true : 1d array-like
        Ground truth (correct) target values.
    y_pred : 1d array-like
        Estimated targets as returned by a classifier.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    Returns
    -------
    balanced_accuracy : float.
        The average_flat of sensitivity and specificity
    See also
    --------
    recall_score
    References
    ----------
    .. [1] Brodersen, K.H.; Ong, alpha.S.; Stephan, K.E.; Buhmann, J.M. (2010).
           The balanced pred and its posterior distribution.
           Proceedings of the 20th International Conference on Pattern Recognition,
           3121–24.
    Examples
    --------
    >>> from decog.metrics import balanced_accuracy_score
    >>> y_true = [0, 1, 0, 0, 1, 0]
    >>> y_pred = [0, 1, 0, 0, 0, 1]
    >>> balanced_accuracy_score(y_true, y_pred)
    0.625
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)

    if y_type != 'binary':
        raise ValueError('Balanced pred is only meaningful '
                         'for binary classification problems.')
    # simply wrap the ``recall_score`` function
    return recall_score(y_true, y_pred,
                        pos_label=None,
                        average='macro',
                        sample_weight=sample_weight)


def _average_multiclass_ovo_score(y_true, y_score, binary_metric=roc_auc_score, average='weighting'):
    """Uses the binary metric for one-vs-one multiclass classification,
    where the score is computed according to the Hand & Till (2001) algorithm.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True multiclass labels.
        Assumes labels have been recoded to 0 to n_classes.
    y_score : array, shape = [n_samples, n_classes]
        Target scores corresponding to probability estimates of a sample
        belonging to a particular class
    average : 'macro' or 'weighted', default='macro'
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean. This does not take label imbalance into account. Classes
            are assumed to be uniformly distributed.
        ``'weighted'``:
            Calculate metrics for each label, taking into account the a priori
            distribution of the classes.
    binary_metric : callable, the binary metric function to use.
        Accepts the following as input
            y_true_target : array, shape = [n_samples_target]
                Some sub-array of y_true for a pair of classes designated
                positive and negative in the one-vs-one scheme.
            y_score_target : array, shape = [n_samples_target]
                Scores corresponding to the probability estimates
                of a sample belonging to the designated positive class label
    Returns
    -------
    score : float
        Average the sum of pairwise binary metric scores
    """
    n_classes = len(np.unique(y_true))
    n_pairs = n_classes * (n_classes - 1) // 2
    prevalence = np.empty(n_pairs)
    pair_scores = np.empty(n_pairs)

    ix = 0
    for a, b in itertools.combinations(range(n_classes), 2):
        a_mask = y_true == a
        ab_mask = np.logical_or(a_mask, y_true == b)

        prevalence[ix] = np.sum(ab_mask) / len(y_true)

        y_score_filtered = y_score[ab_mask]

        a_true = a_mask[ab_mask]
        b_true = np.logical_not(a_true)

        a_true_score = binary_metric(
                a_true, y_score_filtered[:, a])
        b_true_score = binary_metric(
                b_true, y_score_filtered[:, b])
        binary_avg_score = (a_true_score + b_true_score) / 2
        pair_scores[ix] = binary_avg_score

        ix += 1
    return (np.average(pair_scores, weights=prevalence)
            if average == "weighted" else np.average(pair_scores))


def average_multiclass_ovo_score(y_true, y_score, binary_metric=roc_auc_score, average='weighting'):
    """Uses the binary metric for one-vs-one multiclass classification,
    where the score is computed according to the Hand & Till (2001) algorithm.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True multiclass labels.
        Assumes labels have been recoded to 0 to n_classes.
    y_score : array, shape = [n_samples, n_classes]
        Target scores corresponding to probability estimates of a sample
        belonging to a particular class
    average : 'macro' or 'weighted', default='macro'
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean. This does not take label imbalance into account. Classes
            are assumed to be uniformly distributed.
        ``'weighted'``:
            Calculate metrics for each label, taking into account the a priori
            distribution of the classes.
    binary_metric : callable, the binary metric function to use.
        Accepts the following as input
            y_true_target : array, shape = [n_samples_target]
                Some sub-array of y_true for a pair of classes designated
                positive and negative in the one-vs-one scheme.
            y_score_target : array, shape = [n_samples_target]
                Scores corresponding to the probability estimates
                of a sample belonging to the designated positive class label
    Returns
    -------
    score : float
        Average the sum of pairwise binary metric scores
    """
    n_labels = len(np.unique(y_true))
    pos_and_neg_prevalence = []
    label_scores = []
    for pos, neg in itertools.combinations(range(n_labels), 2):
        pos_ix = y_true == pos
        ix = np.logical_or(pos_ix, y_true == neg)

        pos_and_neg_prevalence.append(float(np.sum(ix)) / len(y_true))

        y_score_filtered = y_score[ix]

        class_a = pos_ix[ix]
        class_b = np.logical_not(class_a)

        score_class_a = binary_metric(class_a, y_score_filtered[:, pos])
        score_class_b = binary_metric(class_b, y_score_filtered[:, neg])
        binary_avg_score = (score_class_a + score_class_b) / 2.
        label_scores.append(binary_avg_score)

    # if average_flat == "weighted":
    #     label_scores = np.multiply(np.array(pos_and_neg_prevalence), np.array(label_scores)) / np.mean(pos_and_neg_prevalence)
    #     # label_scores = np.average_flat(label_scores, weights=pos_and_neg_prevalence)
    #     return 2 * np.sum(label_scores) / (n_labels * (n_labels - 1))
    #     # label_scores = np.multiply(np.array(pos_and_neg_prevalence), np.array(label_scores))
    #     # return 2. * np.sum(label_scores) / (n_labels - 1)
    # else:
    #     return 2 * np.sum(label_scores) / (n_labels * (n_labels - 1))
    return np.average(label_scores, weights=pos_and_neg_prevalence) if average == 'weighted' else None

def bacc_ppi(C, alpha=0.05):

    alpha1 = C[0, 0] + 1
    beta1 = C[0, 1] + 1
    alpha2 = C[1, 1] + 1
    beta2 = C[1, 0] + 1

    b_lower = betaavginv(alpha/2, alpha1, beta1, alpha2, beta2)
    b_upper = betaavginv(1 - alpha/2, alpha1, beta1, alpha2, beta2)

    return [b_lower, b_upper]


def bacc_p(C):

    alpha1 = C[0, 0] + 1
    beta1 = C[0, 1] + 1
    alpha2 = C[1, 1] + 1
    beta2 = C[1, 0] + 1

    p = betaavgcdf(0.5, alpha1, beta1, alpha2, beta2)
    return p[0]


def betaavgcdf(x, alpha1, beta1, alpha2, beta2):

    x = np.atleast_1d(x)

    y = betasumcdf(2*x, alpha1, beta1, alpha2, beta2)

    return y

def betaavginv(y, alpha1, beta1, alpha2, beta2):

    # x = fsolve(lambda z: betaavgcdf(z, alpha1, beta1, alpha2, beta2) - y, np.array(0.5))
    x = bisect(lambda z: betaavgcdf(z, alpha1, beta1, alpha2, beta2) - y, 0, 1)

    return x


def betasumcdf(x, alpha1, beta1, alpha2, beta2):

    x = np.atleast_1d(x)

    # Compute the PDF first (since we want the entire pdf rather than just
    # one value from it, using betaconv is computationally more efficient
    # than using betasumpdf)
    res = 0.001
    c = betaconv(res, alpha1, beta1, alpha2, beta2)

    # Sum the PDF up to point x
    y = np.full(x.shape[0], np.nan)
    for i in range(len(x)):
        idx = int(np.round(x[i] / res))
        if idx < 1:
            y[i] = 0
        elif idx > len(c):
            y[i] = 1
        else:
            y[i] = np.trapz(c[:idx]) * res

    return y


def betaconv(res, alpha1, beta1, alpha2, beta2):

    # Set support
    x = np.arange(0, 2+res, res)

    # Individual Beta pdfs
    f1 = beta.pdf(x, alpha1, beta1)
    f2 = beta.pdf(x, alpha2, beta2)

    # Compute convolution
    y = np.convolve(f1, f2)

    # Reduce to [0..2] support
    y = y[:len(x)]

    # Normalize (so that all values sum to 1/res)
    y = y / (np.sum(y) * res)

    return y

def balanced_report(y_true, y_pred, name='', verbose=True):

    score = balanced_accuracy_score(y_true, y_pred)
    C = confusion_matrix(y_true, y_pred)
    CI = bacc_ppi(C, 0.05)
    p = bacc_p(C)
    if verbose:
        print('[%s] Score: %.5f, 95%% CI: [%.5f %.5f] p = %.5f (%.2E)' % (name, 100*score, 100*CI[0], 100*CI[1], p, p))

    return score, CI, p



if __name__ == '__main__':
    y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0]
    C = confusion_matrix(y_true, y_pred)
    # alpha = np.array([[69, 1], [1, 69]])
    print(C)

    ci = bacc_ppi(C, 0.05)
    print('CI: %s' % ci)

    p = bacc_p(C)
    print('p-value: %.50f' % p)

