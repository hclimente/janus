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


def plot_tiles(imgs, emb, nb_y=20, nb_x=20):

    nb_imgs = imgs.shape[0]

    embedding = emb.copy()

    # rescale axes to make things easier
    min_x, min_y = np.min(embedding, axis=0)
    max_x, max_y = np.max(embedding, axis=0)

    embedding[:, 0] = (embedding[:, 0] - min_x) / (max_x - min_x)
    embedding[:, 1] = (embedding[:, 1] - min_y) / (max_y - min_y)
    
    min_y, min_x = np.min(embedding, axis=0)
    max_y, max_x = np.max(embedding, axis=0)

    y_range = np.linspace(min_y, max_y, num=nb_y+1)
    x_range = np.linspace(min_x, max_x, num=nb_x+1)

    s = 1000
    canvas = np.ones((s, s, 3))
    
    img_idx_dict = {}

    for i in range(nb_y):
        for j in range(nb_x):

            idx_x = (x_range[j] <= embedding[:, 0]) & (embedding[:, 0] < x_range[j+1])
            idx_y = (y_range[i] <= embedding[:, 1]) & (embedding[:, 1] < y_range[i+1])

            points = embedding[idx_y & idx_x]

            if len(points) > 0:

                img_idx = np.arange(nb_imgs)[idx_y & idx_x][0]  # take first avilable img in bin
                tile = imgs[img_idx].permute(1, 2, 0)                

                h, w, c = y_range[i + 1] - y_range[i], x_range[j + 1] - x_range[j], 3
                
                delta_y = int(np.around(s * h))
                delta_x = int(np.around(s * w))

                resized_tile = resize(tile, output_shape=(delta_y, delta_x, c))

                y = int(s * y_range[i])
                x = int(s * x_range[j])

                canvas[s - y - delta_y:s - y, x:x + delta_x] = resized_tile
                img_idx_dict[img_idx] = (x, x + delta_x, s - y - delta_y, s - y)

    return canvas, img_idx_dict
