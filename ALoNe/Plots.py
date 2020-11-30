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

def plot_tsne(df, labels=None, show=True):
    palette = sns.color_palette(n_colors=df[labels].max() + 1)
    tsne_plot = sns.scatterplot(data=df,
                                x='tsne 1',
                                y='tsne 2',
                                hue=labels,
                                palette=palette,
                                legend=None,
                                s=10)
    if show:
        plt.show()
    else:
        return tsne_plot

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
