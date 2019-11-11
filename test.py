import numpy as np

trange = np.arange(130)
ws = 5

for t in trange:
    t_min = min(max(0, t-ws), trange.shape[0]-2*ws)
    t_max = max(min(trange.shape[0], t+ws+1), 2*ws)
    print('%g: %s' % (t, range(t_min, t_max)))