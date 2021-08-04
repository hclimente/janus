from functools import lru_cache
from os.path import isfile, join
import random

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
from torch.utils.data import Dataset

from janus.readers import HDF5Reader
from janus.transforms import RandomRot90


class MultiCellDataset(Dataset):
    def __init__(
        self,
        dataset_1,
        dataset_2,
        metadata,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                RandomRot90(),
            ]
        ),
        train_test=False,
    ):

        self.dataset_1 = self.split_train(dataset_1, train_test, 1)
        self.dataset_2 = self.split_train(dataset_2, train_test, 2)
        self.metadata = metadata
        self.transform = transform

    def __getitem__(self, index):

        # give all moas the same weight
        moa1 = random.choice(self.metadata["moa"].unique())
        cell1, moa1 = self.sample_crops(moa1, True, self.dataset_1)

        same_moa = random.getrandbits(1)
        cell2, moa2 = self.sample_crops(moa1, same_moa, self.dataset_2)

        # TODO: If we're loading torch.Tensors rather than PIL.Images,
        #  we should write our own augmentations
        #        if self.transform:
        #            cell1 = self.transform(cell1)
        #            cell2 = self.transform(cell2)

        return cell1, moa1, cell2, moa2, same_moa

    def __len__(self):
        # artificially limit epoch length
        return 10000

    @staticmethod
    def split_train(train, train_test, i, train_size=0.7):

        if train_test:
            train, test = train_test_split(train, train_size=train_size)
            torch.save(test, "test_%s.pkl" % i)
            torch.save(train, "train_%s.pkl" % i)

        return train

    @staticmethod
    def sample_crops(prev_moa, same_moa, dataset):

        moas = tuple([x["moa"] for _, x in dataset])
        idx = np.random.choice(MultiCellDataset.moa_match(moas, prev_moa, same_moa))

        return dataset[idx][0], dataset[idx][1]["moa"]

    @staticmethod
    @lru_cache(maxsize=None)
    def moa_match(moas, moa, same_moa):
        return np.where([same_moa == (m == moa) for m in moas])[0]


class Boyd2019(MultiCellDataset):
    def __init__(
        self,
        data_path,
        metadata,
        padding=32,
        scale=1.0,
        recompute_params=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                RandomRot90(),
            ]
        ),
        train_test=False,
    ):

        mda231 = self.load_crops(
            join(
                data_path,
                "22_384_20X-hNA_D_F_C3_C5_20160031_2016.01.25.17.23.13_MDA231",
            ),
            metadata,
            padding,
            scale,
            recompute_params,
        )
        mda468 = self.load_crops(
            join(
                data_path,
                "22_384_20X-hNA_D_F_C3_C5_20160032_2016.01.25.16.27.22_MDA468",
            ),
            metadata,
            padding,
            scale,
            recompute_params,
        )

        super().__init__(mda231, mda468, metadata, transform, train_test)

    @staticmethod
    def read_metadata(metadata_path):
        metadata = pd.read_excel(metadata_path, engine="openpyxl")

        # remove empty wells
        metadata = metadata[~metadata.content.isnull()]

        # remove wells with no drug/dmso
        metadata = metadata[metadata["content"] != "None"]

        metadata.index = metadata.well

        return metadata

    @staticmethod
    def get_normalization_params(imgs):

        imgs = torch.Tensor.float(imgs)

        avg = torch.mean(imgs, dim=0)
        std = torch.std(imgs, dim=0)

        return avg, std

    @staticmethod
    def load_parameters(pickle_path, crops, recompute_params):

        if recompute_params or not isfile(pickle_path):
            print("computing normalization parameters")
            with open(pickle_path, "wb") as parameters:
                avg, std = Boyd2019.get_normalization_params(crops)
                pickle.dump((avg, std), parameters)
        else:
            with open(pickle_path, "rb") as parameters:
                print("loading normalization parameters")
                avg, std = pickle.load(parameters)

        return avg, std

    @staticmethod
    def load_crops(
        crops_path, metadata, padding, scale=1.0, recompute_params=True, normalize=True
    ):

        crops = list(HDF5Reader.get_crops(crops_path, metadata, padding, scale))

        if normalize:
            avg, std = Boyd2019.load_parameters(
                join(crops_path, "norm_params.pkl"),
                torch.stack([x[0] for x in crops]),
                recompute_params,
            )
            crops = [((x[0] - avg) / std, x[1]) for x in crops]

        return crops


class CellCognition(MultiCellDataset):
    def __init__(
        self,
        data_path,
        metadata,
        train_test=False,
    ):

        mda231 = list(
            HDF5Reader.get_features(
                join(
                    data_path,
                    "22_384_20X-hNA_D_F_C3_C5_20160031_2016.01.25.17.23.13_MDA231",
                ),
                metadata,
            )
        )
        mda231 = [x for x in mda231 if not torch.any(torch.isnan(x[0]))]

        mda468 = list(
            HDF5Reader.get_features(
                join(
                    data_path,
                    "22_384_20X-hNA_D_F_C3_C5_20160032_2016.01.25.16.27.22_MDA468",
                ),
                metadata,
            )
        )
        mda468 = [x for x in mda468 if not torch.any(torch.isnan(x[0]))]

        super().__init__(mda231, mda468, metadata, train_test=train_test)
