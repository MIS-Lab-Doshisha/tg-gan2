import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nilearn.plotting import plot_matrix


# ------ Plot connectivity matrix ------ #
def plot_connectivity_matrix(fig, ax, matrix, labels=None, cmap='bwr', vmin=-1, vmax=1, cbarticksize=6, **kwargs):
    """plot_connectivity_matrix: plot connectivity matrix

    :param fig: matplotlib.pyplot.figure
    :param ax: matplotlib.pyplot.axes
    :param matrix: adjacency matrix (connectivity matrix)
    :param labels: labels of adjacency matrix (connectivity matrix)
    :param cmap: colormap
    :param vmin: minimum value to show
    :param vmax: maximum value to show
    :param cbarticksize: colorbar tick size
    :param kwargs:
    :return: ax: matplotlib.pyplot.axes
    """

    # Plot matrix
    mat = plot_matrix(mat=matrix, labels=labels, axes=ax, colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

    # Setting color bar
    ax_divider = make_axes_locatable(axes=ax)
    cax = ax_divider.append_axes(position='right', size='5%',
                                 pad=0.1)  # setting of an axes to the right of the main axes
    cbr = fig.colorbar(mat, cax=cax)  # add an axes to the  right of the main axes
    cbr.ax.tick_params(labelsize=cbarticksize)

    return ax
