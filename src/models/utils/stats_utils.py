# util functions
# returns series of random values sampled between min and max values of passed col
from sklearn.cluster import KMeans
from numpy.random import random_sample
import pandas as pd
from math import log
def get_rand_data(col):
    rng = col.max() - col.min()
    return pd.Series(random_sample(len(col))*rng + col.min())

def iter_kmeans(df, n_clusters, num_iters=5):
    rng =  range(1, num_iters + 1)
    vals = pd.Series(index=rng)
    for i in rng:
        k = KMeans(n_clusters=n_clusters, n_init=3)
        k.fit(df)
        #print("Ref k: %s" % k.get_params()['n_clusters'])
        vals[i] = k.inertia_
    return vals

def gap_statistic(df, max_k=10, init = 'k-means++'):
    gaps = pd.Series(index = range(1, max_k))
    for k in range(1, max_k):
        if not isinstance(init, str):
            n_init = min(3, init.shape[0])
        else:
            n_init = 3
        km_act = KMeans(n_clusters=k, init = init, n_init=n_init)
        km_act.fit(df)

        # get ref dataset
        ref = df.apply(get_rand_data)
        ref_inertia = iter_kmeans(ref, n_clusters=k).mean()
        intra_inertia = km_act.inertia_
        if ref_inertia == 0 or intra_inertia == 0:
            return gaps[:k]
        gap = log(ref_inertia) - log(intra_inertia)
        gaps[k] = gap

    return gaps