import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
from umap import UMAP


def tsne(x, y):

    x_emb = TSNE().fit_transform(x)
    __plot(x_emb, y, "t-SNE")


def umap(x, y):

    x_emb = UMAP().fit_transform(x)
    __plot(x_emb, y, "UMAP")


def __plot(x, y, algo):

    sns.set_theme()
    g = sns.relplot(x=x[:, 0], y=x[:, 1], hue=y)
    g.set_axis_labels(algo+' 1', algo + ' 2')


def plot_cell(crop):
    crop = crop.permute((1, 2, 0))

    fig, axes = plt.subplots(figsize=(5, 5), ncols=2, nrows=2)

    axes[0][0].imshow(crop[..., 0], cmap='Greys_r')
    axes[0][1].imshow(crop[..., 1], cmap='Greys_r')
    axes[1][0].imshow(crop[..., 2], cmap='Greys_r')
    axes[1][1].imshow(crop)
