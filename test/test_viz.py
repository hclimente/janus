import numpy as np

from viz import tsne, umap


def test_tsne():

    x = np.random.rand(3, 4)
    y = np.random.randint(0, 3, 3)

    tsne(x, y)


def test_umap():

    x = np.random.rand(40, 10)
    y = np.random.randint(0, 3, 40)

    umap(x, y)
