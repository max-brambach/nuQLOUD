import numpy as np
import pandas as pd
import tqdm

import raster_geometry as rg

# TODO: use ellipse from nuclear shape to integrate fluorescence
# TODO: comment functions
# TODO: differentiate nucleus / cytosol / membrane tags
# TODO: add function to __scale__ the intensities // iterate over single samples

def get_intensity_sphere(image, coords, radii, name, disable_statusbar=False):
    coords = np.round(coords).astype(np.int)
    count_error = 0
    if isinstance(radii, int):
        radii = [radii] * coords.shape[0]
    intens_mean = []
    intens_std = []
    for i in tqdm.trange(coords.shape[0], desc='Mapping {} intensities'.format(name), disable=disable_statusbar):
        x, y, z = coords[i, :]
        r = int(radii[i])
        # print(x-r, type(x-r))
        # print(r, type(r))
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

def intensity(df, image, name, mode='nucleus', disable_statusbar=False):
    frames = df['frame'].unique()
    cids = []
    means = []
    stds = []
    for i in frames:
        coords = df[['x', 'y', 'z']].loc[df['frame'] == i].to_numpy()
        mean, std = get_intensity_sphere(image, coords, 2, name=name)
        cids += df['cell id'].loc[df['frame'] == i].tolist()
        means += list(mean)
        stds += list(std)
    df.loc[cids, 'intensity mean ' + name] = means
    df.loc[cids, 'intensity std ' + name] = stds
    return df
