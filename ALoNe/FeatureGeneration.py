import numpy as np
import pandas as pd
import os
import tqdm
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree
from ALoNe.Shape import get_main_vectors, eccentricity, aspect_ratio, ellipsoid_volume, ellipsoid_surface, sphericity


def kernel_density_estimation(p_source, p_target=None):
    """
    Compute the kernel density of a point cloud on a set of discrete points.

    The density field is estimated using the points p_source and it is evaluated at
    the points p_target.
    :param p_source: np.array, contains the x, y, z coordinated of the points used
        to estimate the density field.
        shape: [n, 3], e.g. [[x0, y0, z0], [x1, y1, z1], ... ]
    :param p_target: np.array, contains the x, y, z coordinates of the points used
        to evaluate the estimated density fields.
        Same structure as p_source.
        If none, the density field is evaluated at the source points.
    :return: np.array, contains the estimated source density at the target points.
    """
    if p_target is None:
        p_target = p_source
    kde = gaussian_kde(p_source.T)
    return kde(p_target.T)


def neighbour_density_estimation(p_source, p_target=None, n_neighbours=6):
    """
    Compute the density of a point cloud based on the distance of a point from its n neighbours.

    The k-d tree is constructed using the points p_source and it is queried on the points
    p_target. For each target point the average distance to its n nearest neighbours (source points)
    is returned.
    :param p_source: np.array, contains the x, y, z coordinated of the points used
        to estimate condtruct the k-d tree.
        shape: [n, 3], e.g. [[x0, y0, z0], [x1, y1, z1], ... ]
    :param p_target: np.array, contains the x, y, z coordinates of the points used
        to evaluate the estimated density fields.
        Same structure as p_source.
        If none, the density of the source points is calculated.
    :param n_neighbours: int, number of nearest neighbours over which the distance is averaged.
    :return: np.array, contains the estimated source density at the target points.
    """
    if p_target is None:
        p_target = p_source
    tree = KDTree(p_source)
    dist, ind = tree.query(p_target, k=n_neighbours + 1)
    return 1. / dist[:, 1:].mean(axis=1)


def nuclear_density(df):
    """
    Calculate the kernel and nearest neighbour based density estimates in 3D.

    Convenience function.
    Takes a dataframe as input, which has to have the columns 'x', 'y', 'z' that list the coordinates of the nuclei.
    The returned dataframe is the same as the input dataframe with the columns 'nuclear density kde' and 'nuclear
    density nde' added.
    kde = kernel density estimation, nde = neighbour density estimation
    :param df: pandas dataframe
    :return: pandas dataframe
    """
    coords = df[['x', 'y', 'z']].to_numpy()
    kde = list(kernel_density_estimation(coords))
    nde = list(neighbour_density_estimation(coords))
    df['nuclear density kde'] = np.array(kde) * df['x'].max() * df['y'].max() * df['z'].max()
    df['nuclear density nde'] = nde
    return df


def nuclear_shape(df, disable_status=False):
    """
    Compute nuclear shape features from precision matrices.

    Convenience function.
    Takes a dataframe as input which has to have the following column:
    * 'precision matrix': inverse of the covariance matrix of a multivariate gaussian distribution that was fit to the individual nuclei.

    The input dataframe is returned with the columns added:
    * 'nucleus main vectors cartesian', 'nucleus main vectors spherical': np.arrays containing the major, median and minor vectors (as columns)
    * 'nucleus major axis r', 'nucleus major axis theta', 'nucleus major axis phi': major vector in spherical coordinates
    * 'nucleus eccentricity':
    * 'nucleus aspect ratio':
    * 'nucleus sphericity':
    * 'nucleus volume':
    * 'nucleus surface':
    :param df: pd.Dataframe
    :param disable_status: if True no statusbar is displayed
    :return: pd.Dataframe
    """
    p_matrices = df['precision matrix'].to_numpy()
    c_list = []
    s_list = []
    r_list = []
    theta_list = []
    phi_list = []
    e_list = []
    ar_list = []
    vol_list = []
    surf_list = []
    spher_list = []
    for i in tqdm.trange(len(p_matrices), desc='Calculating nuclear shape', disable=disable_status):
        p_matrix = np.array(p_matrices[i])
        cart_vec, spher_vec = get_main_vectors(p_matrix)
        ecc = eccentricity(spher_vec[0, :])
        c_list.append(cart_vec)
        s_list.append(spher_vec)
        r_list.append(spher_vec[0, 0])
        theta_list.append(spher_vec[1, 0])
        phi_list.append(spher_vec[2, 0])
        e_list.append(ecc)
        ar_list.append(aspect_ratio(spher_vec[0, :]))
        vol_list.append(ellipsoid_volume(spher_vec[0, :]))
        surf_list.append(ellipsoid_surface(spher_vec[0, :]))
        spher_list.append(sphericity(vol_list[-1], surf_list[-1]))
    df['nucleus main vectors cartesian'] = c_list
    df['nucleus main vectors spherical'] = s_list
    df['nucleus major axis r'] = r_list
    df['nucleus major axis theta'] = theta_list
    df['nucleus major axis phi'] = phi_list
    df['nucleus eccentricity'] = e_list
    df['nucleus aspect ratio'] = ar_list
    df['nucleus volume'] = vol_list
    df['nucleus surface'] = surf_list
    df['nucleus sphericity'] = spher_list
    return df


def voronoi_features(df):
    df['boundary bool'] = (df['type'] == 'outside') * 1
    df = get_n_neighbours(df)
    df = voronoi_density(df)
    df = neighbourhood_feature_average(df, 'voronoi volume')
    df = neighbourhood_feature_average(df, 'voronoi sphericity')
    df = neighbourhood_feature_average(df, 'n neighbours')
    df = neighbourhood_feature_average(df, 'boundary bool')
    df = neighbourhood_feature_average(df, 'centroid offset')



def get_n_neighbours(df):
    cids = list(df['cell id'].values)
    n_cids = df['neigbour cell ids'].values
    n_neigh = []
    for i in tqdm.trange(len(n_cids)):
        n_neigh.append([cids[i], len(list(set(cids).intersection(n_cids[i])))])
    n_df = pd.DataFrame(np.array(n_neigh), columns=['cell id', 'n neighbours'])
    df = df.merge(n_df, on='cell id')
    return df


def voronoi_density(df):
    voro_neighbours = df['neigbour cell ids'].to_list()
    idx = df['cell id'].to_dict()
    idx_rev = {v: k for k, v in idx.items()}
    coords = df[list('xyz')].to_numpy()
    out_mean = []
    out_std = []
    for i in tqdm.trange(coords.shape[0]):
        c0 = coords[i, :]
        delta_c = []
        for v in voro_neighbours[i]:
            if v == 0 or v not in df['cell id'].values:
                continue  # neighbour index 0 indicates a boundary and is hence skipped
            c1 = coords[idx_rev[v]]
            delta_c.append(np.linalg.norm(
                c0 - c1))  # THIS COULD BE FURTHER USED AS VECTOR-VALUED FEATURE (without the norm, obviously)
        delta_c = 1 / np.array(delta_c)
        out_mean.append(np.mean(delta_c))
        out_std.append(np.std(delta_c))
    df['density voronoi mean'] = np.array(out_mean)
    df['density voronoi std'] = np.array(out_std)
    return df


def neighbourhood_feature_average(df, feature_name, measures=('mean', 'std')):
    voro_neighbours = df['neigbour cell ids'].to_list()
    idx = df['cell id'].to_dict()
    idx_rev = {v: k for k, v in idx.items()}
    feature = df[feature_name].to_numpy()

    for m in measures:
        out = []
        for i in tqdm.trange(feature.shape[0], desc='neighbourhood {} {}'.format(feature_name, m)):
            f = [feature[i]]
            for v in voro_neighbours[i]:
                if v == 0 or v not in df['cell id'].values:
                    continue  # neighbour index 0 indicates a boundary and is hence skipped
                f1 = feature[idx_rev[v]]
                f.append(f1)
            f = np.array(f)
            if len(f) == 0:
                out.append(np.nan)
                continue
            if m == 'mean':
                out.append(np.mean(f))
            elif m == 'std':
                out.append(np.std(f))
            elif m == 'max':
                out.append(np.max(f))
            else:
                raise ValueError('Unknown value in measures "{}". Used one of [mean, std, max].')
        df['neighbourhood {} {}'.format(feature_name, m)] = np.array(out)
    return df