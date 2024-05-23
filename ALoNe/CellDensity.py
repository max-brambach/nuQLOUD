import numpy as np
import tqdm
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree

"""
Experimental functions for the generation of nuclear density.
Not used, but potentially useful:)
"""

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

def neigbour_density_gradient_estimation(coords, density, n_neighbours=6):
    tree = KDTree(coords)
    dist, ind = tree.query(coords, k=n_neighbours + 1)
    out = []
    for i in tqdm.trange(ind.shape[0]):
        own_density = density[ind[i, 0]]
        neigbour_densities = density[ind[i, 1:]]
        dif_density = neigbour_densities - own_density
        features = [dif_density.mean(), dif_density.std(), dif_density.max(), dif_density.min()]
        out.append(features)
    return np.array(out)

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
    df['nuclear density kde'] = kde
    df['nuclear density nde'] = nde
    return df

def nuclear_density_gradient(df):
    """

    NOT TIME SERIES READY
    :param df:
    :return:
    """
    coords = df[list('xyz')].to_numpy()
    kde = df['nuclear density kde']
    nde = df['nuclear density nde']
    grad_kde = neigbour_density_gradient_estimation(coords, kde)
    grad_nde = neigbour_density_gradient_estimation(coords, nde)
    df['nuclear density gradient mean kde'] = grad_kde[:, 0]
    df['nuclear density gradient std kde'] = grad_kde[:, 1]
    df['nuclear density gradient max kde'] = grad_kde[:, 2]
    df['nuclear density gradient min kde'] = grad_kde[:, 3]
    df['nuclear density gradient mean nde'] = grad_nde[:, 0]
    df['nuclear density gradient std nde'] = grad_nde[:, 1]
    df['nuclear density gradient max nde'] = grad_nde[:, 2]
    df['nuclear density gradient min nde'] = grad_nde[:, 3]
    return df

