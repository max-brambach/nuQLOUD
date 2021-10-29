import numpy as np
import pandas as pd
import tqdm
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler
from ALoNe.Shape import get_main_vectors, eccentricity, aspect_ratio, ellipsoid_volume, ellipsoid_surface, sphericity
import ALoNe


def clean_up_columns(df):
    """
    Remove columns from df that are not used anymore.
    """
    drop_rows = ['cell id TGMM', 'parent id TGMM', 'split score', 'nu', 'beta', 'alpha',
                 'precision matrix', 'vertex number',
                 'edge number', 'edge distance', 'face number', 'voronoi surface area']
    df = df.drop(drop_rows, errors='ignore', axis=1)
    return df

def all_features(df):
    """
    Generate all features from a dataframe.

    Convenience function.
    Batch calculate nuclear density, the restricted voronoi diagram and the corresponding voronoi features for a df
    containing 'x', 'y', 'z' coordinates and 'cell id's (<1).
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    dfs = []
    for sid in df['sample'].unique():
        sdf = df.loc[df['sample'] == sid].copy()
        sdf = ALoNe.FeatureGeneration.multi_scale_density(df)
        sdf = ALoNe.Voronoi.voronoi_restricted(sdf)
        sdf = ALoNe.FeatureGeneration.voronoi_features(sdf)
        dfs.append(sdf)
    df = pd.concat(dfs)
    return df


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

def multi_scale_density(df,
                        radii=None,  # has to be ordered lowest to highest
                        shelled=True,
                        column_name='shell',
                        return_names=True):
    """ Calculate point densities at different scales using a neighbour search on differetly sized spheres.

    :param df: pandas.DataFrame, containing the coordinates of the points in columns lableled 'x', 'y', 'z'
    :param radii: list, containing the radii of the spheres used for the density evaluation. !need to be oredered low->high!
    :param segmented: bool, if True: densities are calculated in sphere shells of volume V=4/3*pi(r_current - r_previous)**3; if False: densities are calculated in full spheres (then smaller radius spheres are contained within larger radius spheres)
    :param column_name: str, prefix of the columns that will be added to the df.
    :param return_names: if True, function returns a list of the column names added to the df.
    """
    if radii is None:
        radii = np.arange(10, 44, 5)
    names = []  # this list will hold the names of the columns that will contain the densities at different radii.
    tree = KDTree(
        df[list('xyz')].values)  # we generate a kdtree from our points to speed up the radial neighbour search.
    cumulative_numel = np.array([0] * len(df.index))
    smallest_r = True
    for r in tqdm.tqdm(radii):  # here we iterate over all radii; i.e. we generate the density for all different radii
        volume = (4 / 3 * np.pi * r ** 3)
        ball = tree.query_ball_tree(tree, r=r)
        numel = []  # this list will hold the number of cells within radius r for each point
        for i in range(len(ball)):
            numel.append(len(ball[i]) - 1)
        numel = np.array(numel)
        if not shelled or smallest_r:
            densities = numel / volume
            smallest_r = False
        else:
            numel = numel - cumulative_numel
            densities = numel / (volume - volume_previous)
        volume_previous = volume
        cumulative_numel += numel
        name = '{} {}'.format(column_name, r)
        df[name] = densities
        names.append(name)
    if return_names:
        return names

def voronoi_features(df):
    df = get_n_neighbours(df)
    df = voronoi_density(df)
    df = neighbourhood_feature_average(df, 'voronoi volume')
    df = neighbourhood_feature_average(df, 'voronoi sphericity')
    df = neighbourhood_feature_average(df, 'n neighbours')
    df = neighbourhood_feature_average(df, 'centroid offset')
    return df



def get_n_neighbours(df):
    cids = list(df['cell id'].values)
    n_cids = df['neigbour cell ids'].values
    n_neigh = []
    for i in tqdm.trange(len(n_cids), desc='number of neighbours'):
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
    for i in tqdm.trange(coords.shape[0], desc='voronoi density', position=0, leave=True):
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


def neighbourhood_feature_average(df, feature_name):
    voro_neighbours = df['neigbour cell ids'].to_list()
    idx = df['cell id'].to_dict()
    idx_rev = {v: k for k, v in idx.items()}
    feature = df[feature_name].to_numpy()
    mean = []
    std = []
    for i in tqdm.trange(feature.shape[0], desc='neighbourhood {}'.format(feature_name), position=0, leave=True):
        f = [feature[i]]
        for v in voro_neighbours[i]:
            if v == 0 or v not in df['cell id'].values:
                continue  # neighbour index 0 indicates a boundary and is hence skipped
            f1 = feature[idx_rev[v]]
            f.append(f1)
        f = np.array(f)
        if len(f) == 0:
            mean.append(np.nan)
            std.append(np.nan)
            continue
        mean.append(np.mean(f))
        std.append(np.std(f))
    df['neighbourhood {} mean'.format(feature_name)] = np.array(mean)
    df['neighbourhood {} std'.format(feature_name)] = np.array(std)
    return df

def scale_features(df, features):
    scaler = StandardScaler()
    X = df[features].to_numpy()
    X_scaled = scaler.fit_transform(X)
    f_scaled = []
    for f in features:
        f_scaled.append(f + ' scaled')
    df_scaled = pd.DataFrame(X_scaled, columns=f_scaled)
    return df.merge(df_scaled, left_index=True, right_index=True), f_scaled