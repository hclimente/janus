import numpy as np
import torch

from viz import tsne, umap, plot_cell


def test_tsne():

    x = np.random.rand(3, 4)
    y = np.random.randint(0, 3, 3)

    tsne(x, y)


def test_umap():

    x = np.random.rand(40, 10)
    y = np.random.randint(0, 3, 40)

    umap(x, y)

def test_plot_cell():

    crop = torch.linspace(0, 4, 5).repeat(3, 5, 1)

    plot_cell(crop)