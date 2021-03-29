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


