import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats
from sklearn.manifold import TSNE
import sklearn.cluster as cls
from sklearn.neighbors import kneighbors_graph
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score
import scipy.cluster.vq
import scipy.spatial.distance
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

# TODO: comment all clustering functions
# TODO: remove unnecessary clustering functions
# TODO: clean up code
# TODO: replace sklearn tsne by opentsne
# TODO: add function to register a dataset to a reference
# TODO: add function to calculate the distance of a point from the cluster center



def linkage_matrix(model):
    """
    Create the linkage matrix of a hierarchical clustering model.

    This can be visualised with scipy.cluster.hierarchy.dendrogram().
    :param model: Output of sklearn.cluster.AgglomerativeClustering().fit().
    :return: np.array()
    """
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    return linkage_matrix

# TOOO: Test the clustering fuctions -- none of them is tested so far 2020-08-06

# def save_data(df,
#               name,
#               features):
#     coords = df[['x', 'y', 'z']].to_numpy()
#     np.save(name + '_coords', coords)
#     X = df[features].to_numpy()
#     np.save(name + '_data.npy', X)
#     return


def scale_data(X,
               mode='min_max'):
    if mode == 'min_max':
        scaler = MinMaxScaler()
    elif mode == 'std':
        scaler = StandardScaler()
    else:
        raise KeyError('Specified mode not supported.\nTry min_max or std.')
    return scaler.fit_transform(X)

def tsne(df,
         features,
         scale_input='min_max'):
    X = df[features].to_numpy()
    if scale_input is not None:
        X = scale_data(X, scale_input)
    n = X.shape[0]
    X_embedded = TSNE(n_components=2,
                      perplexity=n/100,
                      learning_rate=n/12,
                      init='pca',
                      verbose=0).fit_transform(X)
    df['tsne 1'] = X_embedded[:, 0]
    df['tsne 2'] = X_embedded[:, 1]
    return df

def cluster_KMeans(df,
                   features,
                   n_clusters,
                   name=None,
                   scale_input='min_max'):
    X = df[features].to_numpy()
    if scale_input is not None:
        X = scale_data(X, scale_input)
    c = cls.KMeans(n_clusters=n_clusters).fit(X)
    if name is not None:
        df[name] = c.labels_
    else:
        df['cluster labels kmeans'] = c.labels_
    return df


def cluster_hierarchical_agglomerative(df,
                                       features,
                                       connectivity=None,
                                       distance_threshold=None,
                                       n_clusters=8,
                                       return_linkage_matrix=False,
                                       name=None,
                                       scale_input='min_max'):
    X = df[features].to_numpy()
    if scale_input is not None:
        X = scale_data(X, scale_input)


    if connectivity is not None:
        coords = df[['x', 'y', 'z']].to_numpy()
        connectivity = kneighbors_graph(coords, n_neighbors=connectivity)
    c = cls.AgglomerativeClustering(distance_threshold=distance_threshold,
                                    n_clusters=n_clusters,
                                    connectivity=connectivity,
                                    linkage='ward').fit(X)
    if name is not None:
        df[name] = c.labels_
    else:
        df['Cluster Labels Agglomerative'] = c.labels_
    if return_linkage_matrix:
        lm = linkage_matrix(c)
        return df, lm
    else:
        return df


def clustering(df, method, **cluster_parameters):
    """

    :param df:
    :param name:
    :param method:
    :param cluster_parameters:
    :return:
    """
    pass


def estimate_cluster_number_kmeans(df, features,
                                   cluster_range=[2, 20],
                                   scale_input=None,
                                   plot=False):
    X = df[features].to_numpy()
    if scale_input is not None:
        X = scale_data(X, scale_input)
    n_cluster_range = np.arange(cluster_range[0], cluster_range[1])
    list_labels = []
    list_silhouette = []
    list_ellbow = []
    for i in tqdm.trange(n_cluster_range[0], n_cluster_range[-1] + 1):
        model = cls.KMeans(n_clusters=i).fit(X)
        list_labels.append(model.labels_)
        list_silhouette.append(silhouette_score(X, model.labels_))
        list_ellbow.append(model.inertia_)
    list_gap = gap(X, ks=n_cluster_range)
    cn = np.array((np.array(list_ellbow), np.array(list_silhouette), np.array(list_gap))).T
    return cn

def estimate_cluster_number_hierarchical(df, features,
                                         cluster_range=[2, 20],
                                         connectivity=None,
                                         scale_input=None,
                                         ):
    X = df[features].to_numpy()
    if scale_input is not None:
        X = scale_data(X, scale_input)
    n_cluster_range = np.arange(cluster_range[0], cluster_range[1])
    list_labels = []
    list_silhouette = []
    for i in tqdm.trange(n_cluster_range[0], n_cluster_range[-1] + 1):
        model = cls.AgglomerativeClustering(n_clusters=i,
                                            distance_threshold=None,
                                            connectivity=connectivity,
                                            linkage='ward').fit(X)
        list_labels.append(model.labels_)
        list_silhouette.append(silhouette_score(X, model.labels_))
    list_gap = gap(X, ks=n_cluster_range)
    return np.array(list_silhouette), np.array(list_gap), list_labels


def gap(data, refs=None, nrefs=20, ks=range(1, 11)):
    """
    Compute the Gap statistic for an nxm dataset in data.

    Either give a precomputed set of reference distributions in refs as an (n,m,k) scipy array,
    or state the number k of reference distributions in nrefs for automatic generation with a
    uniformed distribution within the bounding box of data.

    Give the list of k-values for which you want to compute the statistic in ks.

    (c) 2013 Mikael Vejdemo-Johansson
    BSD License

    SciPy function to compute the gap statistic for evaluating k-means clustering.
    Gap statistic defined in
    Tibshirani, Walther, Hastie:
         Estimating the number of clusters in a data set via the gap statistic
         J. R. Statist. Soc. B (2001) 63, Part 2, pp 411-423
    """
    dst = scipy.spatial.distance.euclidean
    shape = data.shape
    if refs == None:
        tops = data.max(axis=0)
        bots = data.min(axis=0)
        dists = scipy.matrix(np.diag(tops - bots))

        rands = scipy.random.random_sample(size=(shape[0], shape[1], nrefs))
        for i in range(nrefs):
            rands[:, :, i] = rands[:, :, i] * dists + bots
    else:
        rands = refs

    gaps = np.zeros((len(ks),))
    for (i, k) in enumerate(ks):
        (kmc, kml) = scipy.cluster.vq.kmeans2(data, k)
        disp = sum([dst(data[m, :], kmc[kml[m], :]) for m in range(shape[0])])

        refdisps = np.zeros((rands.shape[2],))
        for j in range(rands.shape[2]):
            (kmc, kml) = scipy.cluster.vq.kmeans2(rands[:, :, j], k)
            refdisps[j] = sum([dst(rands[m, :, j], kmc[kml[m], :]) for m in range(shape[0])])
        gaps[i] = np.log(np.mean(refdisps)) - np.log(disp)
    return gaps