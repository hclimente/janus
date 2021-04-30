import time

import numpy as np
import torch

from janus.viz import tsne, umap, plot_cell, embed_matrix


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


def test_embed_matrix():

    x = np.random.rand(40, 10)

    start = time.time()
    embed_matrix(x, 'umap')
    end = time.time()

    t_first = end - start

    for i in range(10):
        start = time.time()
        embed_matrix(x, 'umap')
        end = time.time()

        t_other = end - start

        assert t_first > t_other
