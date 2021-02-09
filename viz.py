import seaborn as sns
from sklearn.manifold import TSNE
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
