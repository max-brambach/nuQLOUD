import numpy as np
import pandas as pd
import tqdm
from scipy.ndimage import map_coordinates
# from tissue_assembly_pipeline.ImageIO import universal_image_reader
from scipy import signal

# TODO: refactor module name; cell shape is not what we're getting
# TODO: finish ray based analysis
# TODO: use ray based analysis as basis for membrane intensity estimate


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

# -------------- RAY MEMBRANE --------------
def make_fibonacci_lattice(number):
    """
    Generate a spherical fibonacci lattice (Math Geosci (2010) 42: 49–64DOI 10.1007/s11004-009-9257-x)
    with a number of points.
    :param number:
    :return:
    """
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    def z(k, n):
        return -1 + (2 * k) / (n - 1)
    def y(z, k, n):
        return np.sqrt(1 - z**2) * np.sin(2 * np.pi * k * (1 - 1 / phi))
    def x(z, k, n):
        return np.sqrt(1 - z**2) * np.cos(2 * np.pi * k * (1 - 1 / phi))
    def make_coords(k, n):
        Z = z(k, n)
        Y = y(Z, k, n)
        X = x(Z, k, n)
        return np.array([X, Y, Z])
    coords = np.zeros([number, 3])
    for k in range(number):
        coords[k, :] = make_coords(k, number)
    return coords


def make_rays(point, length, number):
    """
    Draw a set of rays that originate from a point.
    :param point:
    :param length:
    :param number:
    :return:
    """
    unit_rays = make_fibonacci_lattice(number)
    shifted_scaled_rays = unit_rays * length + point
    return shifted_scaled_rays



def get_line_profile(image, a, b, line_length, spline_order=1):
    """
    Return the intensities along a line in a 3D image between a and b.
    :param a: [x_a, y_a, z_a]
    :param b: [x_b, y_b, z_b]
    :return: array
    """
    a = np.array(a)
    b = np.array(b)
    # line_length = int(np.round(np.sqrt(np.sum((a-b)**2, axis=0))))
    # if np.any(b < 0) or np.any(np.greater(b, np.array(image.shape) - np.array([1, 1, 1]))):
    #     print('WARNING: Ray goes outside the image.')
    line_x, line_y, line_z = np.linspace(a[0], b[0], line_length), np.linspace(a[1], b[1], line_length), np.linspace(a[2], b[2], line_length)
    profile = map_coordinates(image, np.array([line_x, line_y, line_z]), order=spline_order, mode='constant', cval=0)
    return profile, np.array([line_x.astype(np.int), line_y.astype(np.int), line_z.astype(np.int)]).T

def get_intensity_profiles(image, centre, n_rays, l_rays):
    """
    Return all line profiles of the rays going out from a centre point. Intensity values are read from image.
    :param image:
    :param centre:
    :param n_rays:
    :param l_rays:
    :return:
    """
    rays = make_rays(centre, l_rays, n_rays)
    intensities = np.zeros([n_rays, l_rays])
    coords =  np.zeros([n_rays, l_rays, 3])
    for i in range(n_rays):
        intensities[i, :], coords[i, :, :] = get_line_profile(image=image,
                                     a=centre,
                                     b=rays[i, :],
                                     line_length=l_rays)
    return intensities, coords

def peak_detect(line):
    line = (line - line.mean()) / line.std()
    try:
        peaks, peak_dict = signal.find_peaks(line,
                                             # threshold=0.2,
                                             height=0,
                                             # prominence=1,
                                             # wlen=10,
                                             )
        peaks = np.array(peaks).astype(np.int)
    except ValueError:
        peaks = np.array([])
    return peaks


def cell_shape_rays(coords, image,
                    n_rays=96,
                    l_rays=20,
                    disable_status=False):
    # TODO: function for ray based membrane reconstruction
    # coords = df[['x', 'y', 'z']].to_numpy()
    ray_intensities = np.zeros([coords.shape[0], n_rays, l_rays])
    ray_coords = np.zeros([coords.shape[0], n_rays, l_rays, 3])
    ray_peaks = np.zeros([coords.shape[0], n_rays, 3])
    for i in tqdm.trange(coords.shape[0]):
        c = coords[i, :]
        ray_intensities[i, :, :], ray_coords[i, :, :, :] = get_intensity_profiles(image=image,
                                                                                  centre=c,
                                                                                  n_rays=n_rays,
                                                                                  l_rays=l_rays)
        # print(ray_coords.shape)
        # exit()
        for j in range(n_rays):
            # TODO: calculate ray peaks for each ray and feed them back to the function. A nice function would only return the peak coordinates.
            try:
                peak = peak_detect(ray_intensities[i, j, :])[0]
                ray_peaks[i, j, :] = ray_coords[i, j, int(peak), :]
            except IndexError:
                ray_peaks[i, j, :] = [np.nan]*3
            # print(peak)

            # print(ray_peaks[i, j])
            # exit()
    print(ray_peaks)
    return ray_peaks
    # print(ray_coords.shape)
    # print(ray_intensities.shape)

