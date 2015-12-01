import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class RSLVQ(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes=2, t_alpha=None, t_eta=None, n_prototypes_per_class=1, alpha_crit=0.001, random_state=None, verbose=0):

        t_alpha = 12000 * n_prototypes_per_class if t_alpha is None else t_alpha
        self.t_eta = 10. if t_eta is None else t_eta
        self.n_prototypes_per_class = n_prototypes_per_class
        self.n_prototypes = n_prototypes_per_class * n_classes
        self.alpha_crit = alpha_crit

        alpha0 = 0.1
        self.lr = lambda t_: alpha0 * t_alpha / (t_alpha + t_)

        self.gaussian2 = lambda x_, x0, sigma2_: np.exp(-(np.dot(x_ - x0, x_ - x0)) / (2 * sigma2_))

        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, Y):

        n_samples = len(X)
        sigma2_0 = np.mean(var)
        eta = lambda t_: sigma2_0 * self.t_eta * n_samples / (self.t_eta * n_samples + t_)

        t = 0
        alpha = self.lr(0)
        while alpha > self.alpha_crit:
            alpha = self.lr(t)
            if np.mod(t, 10000) == 0:
                print('alpha(%s): %.4f' % (t, alpha))
            sigma2 = eta(t)
            ind = np.random.randint(self.n_samples)
            x = X[ind]
            y = Y[ind]
            P_denom = np.sum([self.gaussian2(x, prototypes[l], sigma2) for l in range(self.n_prototypes)])
            for l in range(self.n_prototypes):
                P_nom = self.gaussian2(x, prototypes[l], sigma2)
                P = P_nom / (P_denom + 1e-30)
                if prototypes_label[l] == y:
                    if self.n_prototypes_per_class == 1:
                        P_y = 1.
                    else:
                        P_y = P_nom / (np.sum([self.gaussian2(x, prototypes[i], sigma2) for i in range(self.n_prototypes)
                                               if prototypes_label[i] == y]) + 1e-30)
                    prototypes[l] += (alpha / sigma2) * (P_y - P) * (x - prototypes[l])
                else:
                    prototypes[l] -= (alpha / sigma2) * P * (x - prototypes[l])
            t += 1

        return self

    def predict(self, X):

        3