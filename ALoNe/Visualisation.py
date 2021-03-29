import vedo
import vtk
import numpy as np
import seaborn as sns

def show_voro(df, c='gold', alpha=1):
    sourcePoints = vtk.vtkPoints()
    sourcePolygons = vtk.vtkCellArray()
    cells, areas, volumes = [], [], []
    for i in df['cell id']:  # each line corresponds to an input point
        sdf = df.loc[df['cell id'] == i].copy()
        # print(sdf)
        area = sdf['voronoi surface area']
        volu = sdf['voronoi volume']
        n = sdf['vertex number']
        c_vert_relative = np.array(sdf['coordinates vertices'].values[0])
        c_nucleus = sdf[list('xyz')].to_numpy()
        c_vert = c_vert_relative + c_nucleus
        ids = []
        for i in range(c_vert.shape[0]):
            p = c_vert[i, :]
            aid = sourcePoints.InsertNextPoint(p[0], p[1], p[2])
            ids.append(aid)
        vert_per_face = sdf['vertices per face'].values[0]

        faces = []
        for j in range(len(vert_per_face)):
            face = vert_per_face[j]

            ele = vtk.vtkPolygon()
            ele.GetPointIds().SetNumberOfIds(len(face))
            elems = []
            for k, f in enumerate(face):
                ele.GetPointIds().SetId(k, ids[f])
                elems.append(ids[f])
            sourcePolygons.InsertNextCell(ele)
            faces.append(elems)
        cells.append(faces)
        areas.append(area)
        volumes.append(volu)
    poly = vtk.vtkPolyData()
    poly.SetPoints(sourcePoints)
    poly.SetPolys(sourcePolygons)
    voro = vedo.Mesh(poly, c=c, alpha=alpha)
    return voro


def show_categorical_features(df, label, palette='bright'):
    labels = df[label].unique()
    p = vedo.Points(df[list('xyz')].to_numpy(), c='black', alpha=.5, r=1)
    ps = [p]
    colors = sns.palettes.color_palette(palette, n_colors=len(labels))
    for i, l in enumerate(labels):
        coords = df.loc[df[label] == l, list('xyz')].to_numpy()
        p = vedo.Points(coords, c=colors[i], alpha=1, r=3)
        ps.append(p)

    return ps



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


def show_features(df, features, cmap='hot', r=5):
    actors = []
    for f in features:
        pnts = []
        pnts = vedo.Points(df[['x', 'y', 'z']].to_numpy(), r=r)
        pnts.addPointArray(df[f].to_numpy(), f)
        pnts.addScalarBar(title=f)
        pnts.pointColors(cmap=cmap,
                         )
        actors.append(pnts)
    return actors


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