import numpy as np
import pandas as pd
import tqdm
import vedo
from sklearn.mixture import GaussianMixture
import copy

import raster_geometry as rg


def intensity_mapping_gmm(df, images, names):
    out = []
    cids = df['cell id'].unique()

    for i in tqdm.trange(len(cids), position=0, leave=True, desc='intensity mapping {}'.format(names)):
        try:
            cell_id = cids[i]
            temp = [cell_id]
            verts = np.array(df.loc[df['cell id'] == cell_id, 'coordinates vertices'].values[0])
            faces = np.array(df.loc[df['cell id'] == cell_id, 'vertices per face'].values[0], dtype=object)
            coords = np.array(df.loc[df['cell id'] == cell_id, list('xyz')].values[0])
            mesh = vedo.Mesh([verts, faces])
            vol = vedo.volume.mesh2Volume(mesh)
            ar = copy.deepcopy(vol.getDataArray() // 255).astype(bool)
            box_bounds = np.array(list(verts.min(axis=0) + coords - 1) + list(verts.max(axis=0) + coords))
            box_bounds = box_bounds.astype(int)

            for img in images:
                img_slice = copy.deepcopy(img[box_bounds[0]:box_bounds[3],
                                          box_bounds[1]:box_bounds[4],
                                          box_bounds[2]:box_bounds[5]])
                if img_slice.shape != ar.shape:
                    img_slice = img_slice[:ar.shape[0], :ar.shape[1], :ar.shape[2]]
                # voronoi segmentation
                voro_mean = img_slice[ar].mean()
                voro_std = img_slice[ar].std()
                # double gaussian foreground-background classification
                X = img_slice[ar].flatten().reshape(-1, 1)
                gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
                mean1 = gmm.means_[0][0]
                std1 = np.sqrt(gmm.covariances_[0][0][0])
                mean2 = gmm.means_[1][0]
                std2 = np.sqrt(gmm.covariances_[1][0][0])
                labels = gmm.predict(X).astype(bool)
                n1 = (~labels).sum()
                n2 = labels.sum()
                if mean1 < mean2:
                    mean1, mean2 = mean2, mean1
                    std1, std2 = std2, std1
                    n1, n2 = n2, n1
                scaled = mean1 * n1 / (n1 + n2)
                temp += [voro_mean, voro_std, mean1, mean2, std1, std2, n1, n2, scaled]
        except:
            temp += [np.nan]*9
        out.append(temp)
    col_names = ['cell id']
    for name in names:
        col_names += ['intensity {} mean'.format(name),
                      'intensity {} std'.format(name),
                      'intensity {} mean1 gmm'.format(name),
                      'intensity {} mean2 gmm'.format(name),
                      'intensity {} std1 gmm'.format(name),
                      'intensity {} std2 gmm'.format(name),
                      'intensity {} n_voxels1 gmm'.format(name),
                      'intensity {} n_voxels2 gmm'.format(name),
                      'intensity {} scaled gmm'.format(name)]
    df_out = pd.DataFrame(np.array(out), columns=col_names)
    return df.merge(df_out, on='cell id')


def get_intensity_sphere(image, coords, radii, name, disable_statusbar=False):
    coords = np.round(coords).astype(int)
    count_error = 0
    if isinstance(radii, int):
        radii = [radii] * coords.shape[0]
    intens_mean = []
    intens_std = []
    for i in tqdm.trange(coords.shape[0], desc='Mapping {} intensities'.format(name), disable=disable_statusbar):
        x, y, z = coords[i, :]
        r = int(radii[i])
        try:
            sphere = rg.sphere(2*r, r)
            cube = image[x-r:x+r, y-r:y+r, z-r:z+r]
            intens_mean.append(cube[sphere].mean())
            intens_std.append(cube[sphere].std())
        except IndexError:
            try:
                intens_mean.append(image[x, y, z])
                intens_std.append(0)
            except IndexError:
                intens_mean.append(0)
                intens_std.append(0)
                count_error += 1
    if count_error > 0:
        print('Skipped {} cells, because their nuclear position was outside the image volume.'.format(count_error))
    return intens_mean, intens_std

def intensity(df, image, name):
    coords = df[['x', 'y', 'z']].to_numpy()
    mean, std = get_intensity_sphere(image, coords, 2, name=name)
    cids = df.index.tolist()
    means = list(mean)
    stds = list(std)
    df.loc[cids, 'intensity mean ' + name] = means
    df.loc[cids, 'intensity std ' + name] = stds
    return df