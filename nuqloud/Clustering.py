import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.mixture import GaussianMixture
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
import openTSNE as ot
import nuqloud
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

"""
Functions to generate clusters on high dimensional, organisational data and evaluate their performance.
Used in Brambach et al. Not essential for the generation of organisational features.
"""




def scale_data(df, features):
    if not df['cell id'].is_unique:
        df['cell id'] = df.index + 1
    cids_all = df['cell id'].values
    X = df[features].to_numpy()
    X = StandardScaler().fit_transform(X)
    columns = [f + ' scaled' for f in features]
    columns += ['cell id']
    df_scaled = pd.DataFrame(np.hstack([X, cids_all.reshape(-1,1)]), columns=columns)
    return df.merge(df_scaled, on='cell id')


def n_comp_gmm(X, n_comp=(2, 10), n_init=5, plot=True):
    """
    Evaluate the performance of a Gaussian Mixture Model using different metrics.

    Used metrics are the Silhouette Score, the Jensen-Shannon divergence of the split sample population and the
    Bayesian Information Criterion.
    :param X: np.array, shape=[n_observations, n_features], data in sklearn norm
    :param n_comp: list, number of components for the gmm that are tested (n_min, n_max)
    :param n_init: int, number of independent initialisations of the gmm.
    :return: [silhouette_score, js_divergence, bic], each a list with one element per tested number of components.
    """
    s_score = []
    js_distance = []
    bic_score = []
    for i in tqdm.trange(n_comp[0], n_comp[1], desc='number of components for gmm'):
        gmm = GaussianMixture(
            n_components=i,
            covariance_type='full',
            n_init=n_init,
        ).fit(X)
        bic_score.append(gmm.bic(X))
        labels = gmm.predict(X)
        s_score.append(silhouette_score(X, labels))
        X_p, X_q = train_test_split(X, test_size=.5)
        gmm_p = GaussianMixture(
            n_components=i,
            covariance_type='full',
            n_init=n_init,
        ).fit(X_p)
        gmm_q = GaussianMixture(
            n_components=i,
            covariance_type='full',
            n_init=n_init,
        ).fit(X_q)
        js_distance.append(gmm_js(gmm_p, gmm_q, n_samples=X_p.shape[0]//2))
    if plot:
        plt.show(nuqloud.Plots.plot_n_comp_gmm(s_score, bic_score, js_distance))
    return s_score, js_distance, bic_score

def gmm_js(gmm_p, gmm_q, n_samples=20000):
    """
    Jensen-Shannon divergence of two Gaussian Mixture Models.
    :param gmm_p: sklearn.mixture.GaussianMixture object
    :param gmm_q: sklearn.mixture.GaussianMixture object
    :param n_samples: int, subsampling number
    :return: float, js-divergence of the two gmms
    """
    X = gmm_p.sample(n_samples)[0]
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)
    Y = gmm_q.sample(n_samples)[0]
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)
    return (log_p_X.mean() - (log_mix_X.mean() - np.log(2))
            + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2


def tsne_sequential_embedding(df,
                              features,
                              n=25000,
                              name='tsne',
                              initial_perplexities=[50, 500],
                              initial_early_exaggeration=12,
                              initial_exaggeration=1,
                              initial_early_runs=250,
                              initial_runs=750,
                              full_perplexity=30,
                              full_early_exaggeration=12,
                              full_exaggeration=4,
                              full_early_runs=500,
                              full_runs=500):
    """
    Sequential t-sne following the protocol of Kobak & Berens 2019 (https://doi.org/10.1038/s41467-019-13056-x).

    Uses PCA initialisation, subsampling of n=25,000 samples for initial embedding, registration of remaining data and
    final (full) embedding optimisation. The initial embedding uses mulitple perplexities.
    :param df: pd.DataFrame, contains features in columns, observations in rows
    :param features: list, names of the columns in df that contain the features
    :param n:
    :param name:
    :param initial_perplexities:
    :param initial_early_exaggeration:
    :param initial_exaggeration:
    :param initial_early_runs:
    :param initial_runs:
    :param full_perplexity:
    :param full_early_exaggeratio:
    :param full_exaggeration:
    :param full_early_runs:
    :param full_runs:
    :return:
    """
    if len(df.index) <= n:
        n = len(df.index)//2
        Warning('Data contains less samples than were supposed to be used for subsampling.\n'+
                'Using {}, which is half of the data-size.'.format(n))
    initial_df = df.sample(n=n).copy()
    remaining_df = df.loc[~df.index.isin(initial_df.index)].copy()

    # Initial embedding
    affinities_initial = ot.affinity.Multiscale(
        initial_df[features].to_numpy(),
        perplexities=initial_perplexities,
        metric='cosine',
        n_jobs=-1,
        verbose=1,
    )
    init_initial = ot.initialization.pca(
        initial_df[features].to_numpy(),
    )
    initial_embedding = ot.TSNEEmbedding(
        init_initial,
        affinities_initial,
        negative_gradient_method="fft",
        n_jobs=-1,
    )
    initial_embedding = initial_embedding.optimize(
        n_iter=initial_early_runs,
        exaggeration=initial_early_exaggeration,
        momentum=0.5,
        verbose=1,
    )
    initial_embedding = initial_embedding.optimize(
        n_iter=initial_runs,
        exaggeration=initial_exaggeration,
        momentum=0.8,
        verbose=1,
    )
    initial_df['{} 1'.format(name)] = initial_embedding[:, 0]
    initial_df['{} 2'.format(name)] = initial_embedding[:, 1]

    # Registration of the remaining data
    remaining_embedding = initial_embedding.prepare_partial(
        remaining_df[features].to_numpy(),
        k=1,
        perplexities=[1/3],
    )
    remaining_df['{} 1'.format(name)] = remaining_embedding[:, 0]
    remaining_df['{} 2'.format(name)] = remaining_embedding[:, 1]

    # Optimisation of the combined data
    full_df = pd.concat([initial_df, remaining_df])
    affinities = ot.affinity.PerplexityBasedNN(
        full_df[features].to_numpy(),
        perplexity=full_perplexity,
        n_jobs=-1,
        random_state=0,
        verbose=1,
        metric='cosine',
    )
    full_embedding = ot.TSNEEmbedding(
        full_df[['tsne 1', 'tsne 2']].to_numpy(),
        affinities,
        learning_rate=len(full_df.index),
        negative_gradient_method="fft",
        n_jobs=-1,
        verbose=1,
    )
    full_embedding.optimize(
        n_iter=full_early_runs,
        exaggeration=full_early_exaggeration,
        momentum=0.5,
        inplace=True,
    )
    full_embedding.optimize(
        n_iter=full_runs,
        exaggeration=full_exaggeration,
        momentum=0.8,
        inplace=True,
    )
    full_df['{} 1'.format(name)] = full_embedding[:, 0]
    full_df['{} 2'.format(name)] = full_embedding[:, 1]
    return full_df















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
