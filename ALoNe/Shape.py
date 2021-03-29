import numpy as np
import pandas as pd
import tqdm


def normalise_anlges_to_pi(a):
    """
    Limit radiant angles to a range of [0, pi]
    :param a: array, angles
    :return: array
    """
    return a - np.floor_divide(a, np.pi) * np.pi


def eccentricity(a):
    """
    Calculate the eccentricity of an nd ellipsoid by comparing major and minor axis.

    0 = sphere -> 1 maximal elliptical.
    :param a: 1d array, main axis lenghts
    :return: float,
    """
    return np.sqrt(1 - np.min(a)**2 / np.max(a)**2)

def aspect_ratio(a):
    """
    Calculate the aspect ratio of an ellipsoid.
    
    Majort axis / Minor axis
    :param a: 1d array, main axis lenghts
    :return: float
    """
    return np.min(a) / np.max(a)

def ellipsoid_volume(a):
    """
    Calculate the volume of an ellipsoid.

    Use the formula V = a*b*c*pi*3/4, with the length of the main axes a, b, c
    :param a: np.array, length of ellipsoids main axes.
    :return: float, volume of the ellipsoid.
    """
    return np.product(a) * np.pi * 4. / 3.

def ellipsoid_surface(a):
    """
    Calculate the surface of an ellipsoid.

    Use Thompsons's formula http://www.numericana.com/answer/ellipsoid.htm#thomsen
    :param a: np.array, length of ellipsoids main axes.
    :return: float, surface of the ellipsoid.
    """
    p = 1.6075
    return 4.*np.pi*(((a[0]**p *a [1]**p) + (a[0]**p *a [2]**p) + (a[1]**p *a [2]**p))/3.)**(1./p)

def sphericity(volume, surface_area):
    """
    Calculate the sphericity of a 3D object.

    0: not a sphere -> 1: sphere
    :param volume: float, volume of the 3D object.
    :param surface_area: float, surface area of the 3D object.
    :return: float, sphericity of the 3D object
    """
    return np.pi**(1./3.) * (6*volume)**(2./3.) / surface_area

def cart2spher(xyz):
    """
    Convert cartesian vectors to spherical coordinates.

    X, Y, Z -> R, theta, phi
    :param xyz: array (3,n) rows contain x, y, z of cartesian vectors
    :return: format same as input
    """
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[0, :] ** 2 + xyz[1, :] ** 2
    ptsnew[0, :] = np.sqrt(xy + xyz[2, :] ** 2)
    ptsnew[1, :] = np.arctan2(np.sqrt(xy), xyz[2, :])  # for elevation angle defined from Z-axis down
    ptsnew[2, :] = np.arctan2(xyz[1, :], xyz[0, :])
    return ptsnew

def main_vectors(p_matrix):
    """
    Calculate the main vectors from a precision matrix (inverse of covariance matrix).
    
    Use precision matrix of multivariate Gaussian distribution to calculate its main vectors
    (eigenvectors * 1/eigenvalues).
    Since the precision matrix is symmetrical by definition, eigenvalues/-vectors must be real. To avoid complex values
    due to numerical issues (float precision) eigenvalues/vectors are converted to real numbers.
    :param p_matrix: np.array, shape=(n,n), precisison matrix of multivariate Gaussian distribution.
    :return: np.array, shape=(n,n), main vectors of p_matrix, colums are vectors, e.g. [[x0, x1],[y0, y1]]
    """
    eigenvalues, eigenvectors = np.linalg.eig(p_matrix)
    eigenvalues = np.real(np.abs(eigenvalues))
    eigenvectors = np.real(eigenvectors)
    return eigenvectors / np.sqrt(eigenvalues)

def get_main_vectors(p_matrix):
    """
    Return main vectors in cartesian and spherical coordinates from precision matrix.
    Convenience function. It only summarises the functions above.
    :param p_matrix: np.array, shape=(n,n), precision matrix (inverse of covariance matrix).
    :return:
    """
    cart_vec = main_vectors(p_matrix)
    spher_vec = cart2spher(cart_vec)
    spher_vec[1:, :] = normalise_anlges_to_pi(spher_vec[1:, :])
    sort = np.flip(np.argsort(spher_vec[0, :]))
    spher_vec = spher_vec[:, sort]
    cart_vec = cart_vec[:, sort]
    return cart_vec, spher_vec

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


