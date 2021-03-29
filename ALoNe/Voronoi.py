import vedo
import vtk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import tqdm
from scipy.spatial import KDTree


def get_voronoi(ids, coords, boundaries=None):
    """
    Generate Voronoi diagram and some Voronoi cell features using voro++.

    Returns a pd.DataFrame with rows=points and columns:
        * 'cell id', index corresponding to the argument 'ids'
        * 'vertex number', number of vertices
        * 'edge number', number of edges
        * 'edge distance', summed distance of the edges
        * 'face number', number of faces
        * 'voronoi surface area', area of the voronoi cell
        * 'voronoi volume', volume of the voronoi cell
        * 'voronoi sphericity', sphericity of the voronoi cell; 1=sphere, <1 less spherical
        * 'x centroid', 'y centroid', 'z centroid', coordinates of the voronoi cell centroid
        * 'centroid offset', distance between the centroid an the seed point (i.e. coords[i])
        * 'neigbour cell ids', indices of neighbouring cells (neighbours share a face)
        * 'neighbour boundaries', number of neighbours that are a boundary
        * 'coordinates vertices', coordinates of all vertices that belong to the voronoi cell
        * 'vertices per face', indices of the vertices (in 'coordinates vertices') that belong to individual faces
                               of a voronoi cell.
    :param ids: list of ints, indices of the used points (e.g. cell id), NOTE: must be larger than 0, since the values
                              [-1, 0] are associated with specific meaning in voro++.
    :param coords: np.array of shape [n_points, 3(x,y,z)], coordinates of the points
    :param boundaries: list, [x_min, x_max, y_min, y_max, z_min, z_max], boundaries of the cube in which the diagram is
                             generated.
    :return: pd.DataFrame
    """
    if any(ids) < 1:
        raise ValueError('Error in Voronoi diagram generation: ids contain values smaller than 1!')
    with open('temp_coords_for_voronoi.txt', 'w') as f:
        for i, c in enumerate(coords):
            f.write(str(ids[i]) + ' ' + str(c[0]) + ' ' + str(c[1]) + ' ' + str(c[2]) + '\n')
    if boundaries is None:
        boundaries = np.hstack([np.array([0]*3) - 100, coords.max(axis=0) + 100]) # we use 100 since that is much more than the initial voronoi cell size of ~ 45
    boundarystr = str(boundaries[0]) + ' ' + str(boundaries[3]) + ' ' + str(boundaries[1]) + ' ' + str(boundaries[4]) + ' ' + str(boundaries[2]) + ' ' + str(boundaries[5])
    os.system('voro++ -c "%i %w %g %E %s %F %v %c %n %p %t" -v '+boundarystr+' temp_coords_for_voronoi.txt')
    with open('temp_coords_for_voronoi.txt.vol', "r") as f:
        lines = f.readlines()
    rows = []
    for i in tqdm.trange(len(lines), desc='Voronoi cell creation'):
        line = lines[i]
        l = line.strip().split(' ')
        cid = int(l[0])
        n_vert = int(l[1])
        n_edge = int(l[2])
        tot_edge_dist = float(l[3])
        n_face = int(l[4])
        tot_surf_area = float(l[5])
        tot_vol = float(l[6])
        x_centroid = float(l[7])
        y_centroid = float(l[8])
        z_centroid = float(l[9])
        centroid_offset = np.linalg.norm(np.array([x_centroid, y_centroid, z_centroid]))
        nids = []
        bound_neighbours = 0
        for nid in l[10:10+n_face]:
            nids.append(int(nid))
            if int(nid) == 0:
                bound_neighbours += 1
        vertex_coords = []
        for vcoord in l[10+n_face:10+n_face+n_vert]:
            vertex_coords.append(list(np.array(vcoord.replace('(', '').replace(')', '').split(','), dtype=float)))
        vertices_per_face = []
        for vpf in l[10+n_face+n_vert:10+2*n_face+n_vert]:
            vertices_per_face.append(list(np.array(vpf.replace('(', '').replace(')', '').split(','), dtype=int)))
        sphericity = (np.pi**(1/3) * (6 * tot_vol)**(2/3) )/(tot_surf_area)
        rows.append([cid, n_vert, n_edge, tot_edge_dist, n_face, tot_surf_area, tot_vol, sphericity, x_centroid, y_centroid, z_centroid,
                   centroid_offset, nids, bound_neighbours, vertex_coords, vertices_per_face])
    col_names = ['cell id', 'vertex number', 'edge number', 'edge distance', 'face number', 'voronoi surface area', 'voronoi volume', 'voronoi sphericity',
                'x centroid', 'y centroid', 'z centroid', 'centroid offset', 'neigbour cell ids', 'neighbour boundaries', 'coordinates vertices', 'vertices per face']
    df_out = pd.DataFrame(data=rows, columns=col_names)
    return df_out


def volume2radius(volume):
    """
    Calculate the radius of a sphere using its volume.
    :param volume: float
    :return: float
    """
    return (3/4/np.pi*volume)**(1/3)


def make_fibonacci_lattice(number):
    """
    Generate a spherical fibonacci lattice (Math Geosci (2010) 42: 49–64DOI 10.1007/s11004-009-9257-x)
    with a number of points.
    :param number: int, number of points
    :return: np.array, coordinates of the fibonacci lattice points on the unit sphere in cartesian coordinates.
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
    return coords.astype(float)


def adaptive_radial_restriction_3d(df, k=10, flat=False, flat_scale=.66):
    """
    Generate a set of points that will adaptively restrict a voronoi diagram relative to the local arrangement of the
    seed points.

    Restriction points are initialised equidistantly on a sphere with the average radius of the voronoi neighbourhood
    and kept, if they are contained within the seeds voronoi cell.
    :param df: pd.DataFrame, containing the columns:
                                * 'x', 'y', 'z': seed coordinates
                                * 'boundary bool': True = seed at boundary, False = seed inside volume
                                * 'cell id': int, unique number that identifies a seed
                                * 'neighbour cell ids': cell ids of neighbouring cells
    :param k: int, number of nearest neighbours off of which the regularisation parameters are determined if all voronoi
                   neighbours are boundaries. If all k nearest neighbours are still boundaries, this is repeated for
                   k+5 neighbours until non-boundary cells are found.
    :return: np.array, shape=[n_points, 3(coords)]
    """
    pnts = df[list('xyz')]
    tree = KDTree(pnts)
    restriction_points = []
    df_bound = df.loc[df['boundary bool'] == 1].copy()
    cids = df['cell id'].values
    df_idx = df.index
    dict_cid_idx = dict(zip(cids, df_idx))
    pbar = tqdm.tqdm(total=len(df_bound['cell id'].index), desc='Adaptive radial restriction')
    for cid in df_bound['cell id']:
        pbar.update(1)
        try:
            nid = np.array(df.loc[df['cell id'] == cid]['neigbour cell ids'])[0]
            df_neigh = df.loc[df['cell id'].isin(nid)]
            bound = df_neigh['boundary bool'].values.astype(bool)
            if bound.all():
                if not flat:
                    k_plus = 0
                    while bound.all():
                        kdtree_neighbours = tree.query(df.loc[df['cell id'] == cid, list('xyz')].values[0],
                                                       k=k + k_plus)[1][1:]
                        df_neigh = df.iloc[kdtree_neighbours]
                        bound = df_neigh['boundary bool'].values.astype(bool)
                        k_plus += 5
                else:
                    coords_neigh = df_neigh[list('xyz')].values
                    coords_seed = df.loc[df['cell id'] == cid, list('xyz')].values
                    avg_dist = np.mean(np.linalg.norm(coords_neigh - coords_seed, axis=1)) * flat_scale
            V = df_neigh['voronoi volume'].values
            n_vert = df_neigh['vertex number'].values
            V = V[~bound].mean()
            radius = volume2radius(V)
            n_vert = int(np.round(np.mean(n_vert[~bound])))
            r_points = make_fibonacci_lattice(n_vert) * radius * 2 + df.loc[df['cell id'] == cid][list('xyz')].values
            nearest_pnt_idx = tree.query(r_points)[1]
            if any(nearest_pnt_idx == dict_cid_idx[cid]):
                restriction_points.append(r_points[nearest_pnt_idx == dict_cid_idx[cid]].astype(float))
        except IndexError:
            print('skipping {}'.format(cid))
    return np.vstack(restriction_points)


def find_boundary_cells(df):
    cids = df['cell id'].values
    df_idx = df.index
    dict_cid_idx = dict(zip(cids, df_idx))
    n_cids = df['neigbour cell ids'].values
    rest_bool = df['restriction point bool'].values
    rest_cids = df.loc[df['restriction point bool'] == True, 'cell id'].values
    t_list = []
    for i, n_cid in enumerate(n_cids):
        if rest_bool[i]:
            t_list.append('restriction')
        elif any(list(set(n_cid).intersection(rest_cids))):
            t_list.append('boundary')
        else:
            t_list.append('inside')
    df['point type'] = t_list
    return df


def voronoi_restricted(df, flat=False):
    voro1 = get_voronoi(df['cell id'], df[list('xyz')].values)
    df = pd.merge(df, voro1, on='cell id')
    df['boundary bool'] = (df['neighbour boundaries'] > 0) * 1
    p_arr = adaptive_radial_restriction_3d(df, flat=flat)
    df_arr = pd.DataFrame(p_arr, columns=list('xyz'))
    df_arr['restriction point bool'] = True
    df['restriction point bool'] = False
    df_full = pd.concat([df, df_arr])
    df_full.reset_index(inplace=True, drop=True)
    df_full['cell id'] = df_full.index + 1
    df_full.drop(['vertex number', 'edge number', 'edge distance', 'face number', 'voronoi surface area',
                  'voronoi volume', 'voronoi sphericity', 'x centroid', 'y centroid', 'z centroid', 'centroid offset',
                  'neigbour cell ids', 'neighbour boundaries', 'coordinates vertices', 'vertices per face'],
                 axis=1, inplace=True)
    voro_full = get_voronoi(df_full['cell id'], df_full[list('xyz')].values)
    df_full = pd.merge(df_full, voro_full, on='cell id')
    df_full = find_boundary_cells(df_full)
    df_full = df_full.loc[df_full['restriction point bool'] != True]
    df_full.drop(['boundary bool', 'restriction point bool'], axis=1, inplace=True)
    return df_full


