import vtkplotter as vtp
import numpy as np
import pandas as pd
import tqdm
import seaborn as sns

# TODO: clean up and comment
# TODO: make cmap flexible
# TODO: remove frame arguments
# TODO: remove unnecessary functions (i.e. tracking stuff)


def show_tracks(df, track_ids=None, disable_statusbar=False):
    if track_ids is None:
        track_ids = np.arange(df['track id'].max())
    plot_list = []
    pbar = tqdm.tqdm(total=len(plot_list), desc='Building track objects', disable=disable_statusbar)
    for tid in track_ids:
        s_list = []
        d_list = []
        track = df.loc[df['track id'] == tid]
        for i, node in track.iterrows():
            if node['parent id'] == -1:
                continue
            source_coords = track.loc[track['cell id'] == node['parent id']][['x', 'y', 'z']].to_numpy().squeeze()
            destination_coords = node[['x', 'y', 'z']].to_numpy()
            s_list.append(source_coords)
            d_list.append(destination_coords)
        if len(s_list) == 0 or len(d_list) == 0:
            continue
        # print('s_list: ', s_list)
        # print('d_list: ', d_list)
        track_plot_object = vtp.Lines(s_list, d_list, c=i)
        plot_list.append(track_plot_object)
        pbar.update(1)
    vtp.show(plot_list, bg='k')


def show_clusters(df, labels, mode, export=None, bg='w'):
    coords = df[['x', 'y', 'z']].to_numpy()
    labels = df[labels].to_numpy()
    pnts = vtp.Points(coords, r=1, c='gray', alpha=.5)
    labels_list = np.unique(labels)
    actors = [pnts]
    palette = sns.color_palette(n_colors=len(labels_list))
    if mode == 'points':
        for l in labels_list:
            actors.append(vtp.Points(coords[labels==l], r=2, c=palette[l]))
    if mode == 'mesh':
        for l in labels_list:
            msh = vtp.convexHull(coords[labels==l], alphaConstant=30).color(palette[l])
            msh.smoothMLS2D()
            msh.fillHoles()
            actors.append(msh)
    vtp.show(actors, bg=bg)
    if export is not None:
        vtp.exportWindow(export)

def show_orientation(df, frame):
    pass


def show_features(df, frame, features, export=None):
    actors = []
    df = df.loc[df['frame'] == frame]
    for f in features:
        pnts = []
        pnts = vtp.Points(df[['x', 'y', 'z']].to_numpy(), r=5)
        vals = df[f].to_numpy() / df[f].max()
        pnts.addPointArray(vals, f)
        pnts.addScalarBar(title=f)
        pnts.pointColors(cmap='hot',
                         )
        actors.append(pnts)
    vtp.show(actors, N=len(actors))
    if export is not None:
        vtp.exportWindow(export)

def show_nuclear_fit(df, cid):
    pass
    # TODO integrate code below, such that nuclear fit can be verified
    # difference = gmm_out - cube
    # fig = plt.figure()
    # ax1 = plt.subplot(331)
    # p1 = ax1.imshow(cube.max(axis=0))
    # ax2 = plt.subplot(332)
    # p2 = ax2.imshow(gmm_out.max(axis=0))
    # ax3 = plt.subplot(333)
    # p3 = ax3.imshow(difference.max(axis=0))
    # ax4 = plt.subplot(334)
    # p1 = ax4.imshow(cube.max(axis=1))
    # ax5 = plt.subplot(335)
    # p2 = ax5.imshow(gmm_out.max(axis=1))
    # ax6 = plt.subplot(336)
    # p3 = ax6.imshow(difference.max(axis=1))
    # ax7 = plt.subplot(337)
    # p1 = ax7.imshow(cube.max(axis=2))
    # ax8 = plt.subplot(338)
    # p2 = ax8.imshow(gmm_out.max(axis=2))
    # ax9 = plt.subplot(339)
    # p3 = ax9.imshow(difference.max(axis=2))
    # fig.show()