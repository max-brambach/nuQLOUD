import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# TODO: refactor to 'plot'
# TODO: comment and clean up
# TODO: us **kwargs to make plot functions more flexible


def num2grid(n):
    """
    Calculate the dimensions a grid that can contain at least n elements.

    The grid is defined such that the number of rows and columns are both minimal.

    :param n: int, number of elements the grid has to contain
    :return: (r, c), number of rows and columns that define the grid.
    """
    r = np.floor(np.sqrt(n))
    c = np.ceil(n/r)
    return int(r), int(c)


def plot_histograms(df, quantities):
    # TODO: add documentaion to plot_histograms
    # TODO: add fuction that plots histograms of all relevant quantities
    rows, cols = num2grid(len(quantities))
    plt.figure(figsize=(4*cols, 4* rows))
    for i, quant in enumerate(quantities):
        ax = plt.subplot(rows, cols, i+1)
        sns.distplot(df[quant],ax=ax)
    plt.show()

def plot_tsne(df,
              tsne_labels=['tsne 1', 'tsne 2'],
              hue='cluster labels',
              figsize=[10,10],
              s=10,
              palette='bright',
              linewidth=0,
             ):
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df,
                   x=tsne_labels[0],
                   y=tsne_labels[1],
                   hue=hue,
                   palette=palette,
                   legend=None,
                   s=s,
                   linewidth=linewidth)
    return plt.gca()

def plot_n_comp_gmm(s_score, bic_score, js_distance):
    f = plt.figure(figsize=(15,5))
    ax = plt.subplot(1, 3, 1)
    ax.set_title('silhouette score')
    plt.bar(x=range(2,10),
                height=s_score,
                )
    ax = plt.subplot(1, 3, 2)
    ax.set_title('bayesian information criterion ')
    plt.bar(x=range(2,10),
                height=bic_score,
                )
    ax = plt.subplot(1, 3, 3)
    ax.set_title('Jensen-Shannon divergence')
    plt.bar(x=range(2,10),
                height=js_distance,
                )
    return f


def plot_cluster_number_estimation(cn):
    sns.set(font_scale=1.5, style='whitegrid')
    palette = sns.color_palette(n_colors=3)
    plt.figure(figsize=(15, 5))
    title = ['ellbow', 'silhouette score', 'gap statistic']
    for i in range(cn.shape[1]):
        ax = plt.subplot('13{}'.format(1 + i))
        ax.plot(cn[:, i], linewidth=5, color=palette[i])
        ax.set_title(title[i])
        ax.set_xlabel('n clusters')
        # ax.grid()
        ax.set_xticks(np.arange(0, 20, 4))
        ax.set_xticklabels(np.arange(2, 22, 4))
        ax.set_yticklabels([])
    plt.show()
