import pandas as pd
import numpy as np
import tqdm
import sys
from scipy.spatial import KDTree
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
sys.path.append(r'C:\Users\MaxB\PycharmProjects\TissueAssemblyPipeline\scripts')

from tissue_assembly_pipeline.CellShape import eccentricity, aspect_ratio, ellipsoid_volume, ellipsoid_surface, sphericity, cart2spher, normalise_anlges_to_pi

# TODO: _old_ make a function that extracts the region around a nucleus from an image. this couls be done by next neighbour/2 distance
# TODO: _old_ fit ellipsoid on data: http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html, https://vtkplotter.embl.es/_modules/vtkplotter/analysis.html pca ellipsoid

# TODO: scavenge the NuclearShape.py module for useful functions and delete



def img_to_coordlist(img):
    x, y, z = np.indices(img.shape)
    coords = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    out = []
    for i in range(coords.shape[0]):
        x, y, z = coords[i, :]
        out += [x, y, z] * img[x, y, z]
    out = np.array(out).reshape((-1, 3))
    return out

def fit_3d_gaussian_to_image(img, mean):
    if mean.shape == (3,):
        mean = np.column_stack([mean, mean, mean])
    img = np.round(np.clip(img, a_min=img.mean()+img.std()/2, a_max=None) - img.mean()-img.std()/2).astype(int)
    data = img_to_coordlist(img)
    gmm = GaussianMixture(n_components=3, covariance_type='tied', init_params='random', means_init=mean)
    gmm.fit(data)
    return gmm

def covariance_matrix_to_ellipsoid(cov, confidence_level=95):
    """

    magnification factor from confidence level https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4358977/
    :param cov:
    :param confidence_level:
    :return:
    """
    mag_fact_from_conf_level = {80: 2.1544, 85: 2.3059, 90: 2.5003, 95: 2.7955, 99: 3.3682, 99.9: 4.0331}
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    eigenvalues = mag_fact_from_conf_level[confidence_level] * np.sqrt(np.real(np.abs(eigenvalues)))
    eigenvectors = np.real(eigenvectors)
    return eigenvectors * eigenvalues

def get_main_vectors_from_cov(cov):
    """
    Return main vectors in cartesian and spherical coordinates from precision matrix.
    Convenience function. It only summarises the functions above.
    :param cov: np.array, shape=(n,n), covariance matrix.
    :return:
    """
    cart_vec = covariance_matrix_to_ellipsoid(cov, confidence_level=95)
    spher_vec = cart2spher(cart_vec)
    spher_vec[1:, :] = normalise_anlges_to_pi(spher_vec[1:, :])
    sort = np.flip(np.argsort(spher_vec[0, :]))
    spher_vec = spher_vec[:, sort]
    cart_vec = cart_vec[:, sort]
    # print(cart_vec)
    # print(spher_vec)
    return cart_vec, spher_vec

def nuclear_shape_from_image(df, img, verbose=False, disable_statusbar=False):
    coords = df[['x', 'y', 'z']].to_numpy()
    cids = df['cell id'].to_numpy()
    tree = KDTree(coords)
    distances, neighbours = tree.query(coords, 2)
    radii = np.ceil(distances[:, 1]/2).astype(int)
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
    cov_list = []
    new_coords_list = []
    pbar = tqdm.tqdm(total=len(cids)+1, desc='Generating nuclear shapes', disable=disable_statusbar)
    coords = np.round(coords).astype(int)
    for cid in cids:
        x, y, z = coords[cid, :]
        r = radii[cid]
        cube = img[x-r:x+r, y-r:y+r, z-r:z+r]
        x, y, z = np.mgrid[0:cube.shape[0]:1, 0:cube.shape[1]:1, 0:cube.shape[2]:1]
        pos = np.column_stack((x.flat, y.flat, z.flat))
        gmm = fit_3d_gaussian_to_image(img=cube, mean=np.array([r, r, r]))
        cart_vec, spher_vec = get_main_vectors_from_cov(gmm.covariances_)
        ecc = eccentricity(spher_vec[0, :])
        new_coords = coords[cid, :] - r + gmm.means_.mean(axis=1)
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
        cov_list.append(gmm.covariances_)
        new_coords_list.append(new_coords)
        if verbose:
            print('\nCELL ID: {}\n------'.format(cid))
            print('main axes length: ', spher_vec[0, :])
            print('eccentricity: ', e_list[-1])
            print('aspect ratio: ', ar_list[-1])
            print('sphericity: ', spher_list[-1])
            print('volume: ', vol_list[-1])
            print('surface: ', surf_list[-1])
            print('coords: ', new_coords_list[-1])
        pbar.update()
    # cids = cid
    print(len(c_list), len(cids))
    new_coords_list = np.vstack(new_coords_list)
    # df.loc[cids, 'nucleus main vectors cartesian'] = c_list
    # df.loc[cids, 'nucleus main vectors spherical'] = s_list
    df.loc[cids, 'nucleus major axis r'] = r_list
    df.loc[cids, 'nucleus major axis theta'] = theta_list
    df.loc[cids, 'nucleus major axis phi'] = phi_list
    df.loc[cids, 'nucleus eccentricity'] = e_list
    df.loc[cids, 'nucleus aspect ratio'] = ar_list
    df.loc[cids, 'nucleus volume'] = vol_list
    df.loc[cids, 'nucleus surface'] = surf_list
    df.loc[cids, 'nucleus sphericity'] = spher_list
    # df.loc[cids, 'covariance matrix fit'] = cov_list
    df.loc[cids, 'x fit'] = new_coords_list[:, 0]
    df.loc[cids, 'y fit'] = new_coords_list[:, 1]
    df.loc[cids, 'z fit'] = new_coords_list[:, 2]
    # df[cids, 'x']
    pbar.update()
    return df
#         add cov matrix
#         add new coord
#         add all measures

def nuclear_alignment_local(df, n_neighbours=6, desc='Generating local alignment', disable_statusbar=False):
    cids = df['cell id'].to_numpy()
    coords = df[['x', 'y', 'z']].to_numpy()
    tree = KDTree(coords)
    distances, neighbours = tree.query(coords, 1+n_neighbours)
    theta = df['nucleus major axis theta'].to_numpy()
    phi = df['nucleus major axis phi'].to_numpy()
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    vectors = np.column_stack([x, y, z])
    allign = []
    for i in tqdm.trange(len(x), desc='Generating nuclear alignment', disable=disable_statusbar):
        al = 0
        for j in range(n_neighbours):
            al += abs(np.dot(vectors[i, :], vectors[neighbours[i, j+1]]))
        allign.append(al/n_neighbours)
    df.loc[cids, 'local alignment'] = allign
    return df

