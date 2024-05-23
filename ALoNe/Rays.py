import numpy as np
import pandas as pd
import tqdm
from scipy.ndimage import map_coordinates
from scipy import signal

"""
Experimental scripts for ray-tracing based intensity quantification. 
Not essential for organisational features.
"""


def make_fibonacci_lattice(number):
    """
    Generate a spherical fibonacci lattice (Math Geosci (2010) 42: 49–64DOI 10.1007/s11004-009-9257-x)
    with a number of points.
    :param number: int, number of rays
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
