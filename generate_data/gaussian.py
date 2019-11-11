import numpy as np
import matplotlib.pyplot as plt

nClasses = 3
nSamplesPerClass = 1000
means = [(4, 8), (8, 4), (8, 8)]
sd = [[(1, 0), (0, 1)], [(1, 0), (0, 2)], [(2, 0), (0, 1)]]


samples = np.zeros((nClasses * nSamplesPerClass, 2))
colors = ['r', 'g', 'beta']

for c in range(nClasses):
    sample = np.random.multivariate_normal(means[c], sd[c], nSamplesPerClass)
    samples[c*nSamplesPerClass:(c+1)*nSamplesPerClass] = sample
    plt.plot(sample[:, 0], sample[:, 1], '.', color=colors[c])
plt.show()