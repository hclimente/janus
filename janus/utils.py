import torch

from janus.datasets import Boyd2019, CellCognition, MultiCellDataset


def split_by_well(data, metadata, seed, padding, scale, transforms):
    tr_metadata = metadata.sample(
        frac=0.7, weights=metadata.groupby("moa")["moa"].transform("count")
    )
    tr_metadata.to_csv("tr_seed_%s.tsv" % seed, sep="\t", index=False)

    tr_data = Boyd2019(
        data,
        tr_metadata,
        padding=padding,
        scale=scale,
        transform=transforms,
    )

    te_metadata = metadata.drop(tr_metadata.index)
    te_metadata.to_csv("te_seed_%s.tsv" % seed, sep="\t", index=False)

    te_data = Boyd2019(
        data, te_metadata, padding=padding, scale=scale, transform=transforms
    )

    return tr_data, te_data


def split_by_crop(data, metadata, padding, scale, transforms):
    tr_data = Boyd2019(
        data,
        metadata,
        padding=padding,
        scale=scale,
        train_test=True,
        transform=transforms,
    )

    te_1 = torch.load("test_1.pkl")
    te_2 = torch.load("test_2.pkl")

    te_data = MultiCellDataset(te_1, te_2, metadata, transform=transforms)

    return tr_data, te_data


def split_features_by_well(data, metadata, seed):

    tr_metadata = metadata.sample(
        frac=0.7, weights=metadata.groupby("moa")["moa"].transform("count")
    )
    tr_metadata.to_csv("tr_seed_%s.tsv" % seed, sep="\t", index=False)

    tr_data = CellCognition(data, tr_metadata)

    te_metadata = metadata.drop(tr_metadata.index)
    te_metadata.to_csv("te_seed_%s.tsv" % seed, sep="\t", index=False)

    te_data = CellCognition(data, te_metadata)

    return tr_data, te_data
