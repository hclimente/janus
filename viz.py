from functools import lru_cache

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from umap import UMAP

import numpy as np
from skimage.transform import resize


def tsne(x, y):

    x_emb = get_embedding(x, 'tsne')
    xy_plot(x_emb, y, "t-SNE")


def umap(x, y):

    x_emb = get_embedding(x, 'umap')
    xy_plot(x_emb, y, "UMAP")


def xy_plot(x, y, emb='Dimension'):

    sns.set_theme()
    g = sns.relplot(x=x[:, 0], y=x[:, 1], hue=y)
    g.set_axis_labels(emb + ' 1', emb + ' 2')


def get_embedding(x, emb):

    return __get_embedding(tuple(tuple(e) for e in x), emb)


@lru_cache(maxsize=None)
def __get_embedding(x, emb):

    if emb == 'tsne':
        x_emb = TSNE().fit_transform(x)
    elif emb == 'umap':
        x_emb = UMAP().fit_transform(x)

    return x_emb


def plot_cell(crop):
    crop = crop.permute((1, 2, 0))

    fig, axes = plt.subplots(figsize=(5, 5), ncols=2, nrows=2)

    axes[0][0].imshow(crop[..., 0], cmap='Greys_r')
    axes[0][1].imshow(crop[..., 1], cmap='Greys_r')
    axes[1][0].imshow(crop[..., 2], cmap='Greys_r')
    axes[1][1].imshow(crop)


def plot_tiles(imgs, emb, grid_units=50, pad=1):
    
    # roughly 1000 x 1000 canvas
    cell_width = 1000 // grid_units
    s = grid_units * cell_width

    nb_imgs = imgs.shape[0]

    embedding = emb.copy()

    # rescale axes to make things easier
    min_x, min_y = np.min(embedding, axis=0)
    max_x, max_y = np.max(embedding, axis=0)

    embedding[:, 0] = s * (embedding[:, 0] - min_x) / (max_x - min_x)
    embedding[:, 1] = s * (embedding[:, 1] - min_y) / (max_y - min_y)

    canvas = np.ones((s, s, 3))
    
    img_idx_dict = {}

    for i in range(grid_units):
        for j in range(grid_units):

            idx_x = (j * cell_width <= embedding[:, 1]) & (embedding[:, 1] < (j + 1) * cell_width)
            idx_y = (i * cell_width <= embedding[:, 0]) & (embedding[:, 0] < (i + 1) * cell_width)

            points = embedding[idx_y & idx_x]

            if len(points) > 0:

                img_idx = np.arange(nb_imgs)[idx_y & idx_x][0]  # take first avilable img in bin
                tile = imgs[img_idx].permute(1, 2, 0)                

                
                resized_tile = resize(tile, output_shape=(cell_width - 2 * pad, cell_width - 2 * pad, 3))

                y = j * cell_width
                x = i * cell_width

                canvas[s - y - cell_width+pad:s - y - pad, x + pad:x+cell_width - pad] = resized_tile
                img_idx_dict[img_idx] = (x, x + cell_width, s - y - cell_width, s - y)

    return canvas, img_idx_dict
