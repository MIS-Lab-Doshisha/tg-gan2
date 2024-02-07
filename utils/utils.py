import re
import scipy.io as io
from matplotlib.colors import ListedColormap


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def load_cmap_mat(cmap_filename):

    matdata = io.loadmat(file_name=cmap_filename)
    cdata = matdata['map']
    newcmp = ListedColormap(cdata)

    return newcmp


