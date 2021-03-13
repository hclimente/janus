from functools import partial
import random

import pandas as pd
from torchvision import transforms
import torch
from torch.utils.data import Dataset

from readers import HDF5Reader
from transforms import RandomRot90


class MultiCellDataset (Dataset):

    def __init__(self, dataset_1, dataset_2, metadata, transform):

        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.metadata = metadata
        self.transform = transform

    def __getitem__(self, index, balanced=True):
        random.seed(index)

        if balanced:
            moa1 = random.choice(self.metadata['moa'].unique)
            cell1, moa1 = self.sample_crops(moa1, True, self.dataset_1)
        else:
            cell1, moa1 = next(self.dataset_1())

        same_moa = random.getrandbits(1)
        cell2, moa2 = self.sample_crops(moa1, same_moa, self.dataset_2)

        if self.transform:
            cell1 = self.transform(cell1)
            cell2 = self.transform(cell2)

        return cell1, moa1, cell2, moa2, same_moa

    def __len__(self):
        # artificially limit epoch length
        return 100000

    @staticmethod
    def sample_crops(prev_moa, same_moa, dataset):
        while True:
            crop, info = next(dataset())

            if same_moa and prev_moa == info['moa']:
                break
            elif not same_moa and prev_moa != info['moa']:
                break

        return crop, info['moa']


class Boyd2019(MultiCellDataset):

    def __init__(self, path_metadata,
                 transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.RandomVerticalFlip(),
                                               RandomRot90()])):

        padding = 32
        self.metadata = self.read_metadata(path_metadata)

        self.avg_mda231 = None
        self.std_mda231 = None
        self.avg_mda468 = None
        self.std_mda468 = None

        # wrap iterators in partial to get a fresh call
        mda231_crops = HDF5Reader.get_crops('test/data/22_384_20X-hNA_D_F_C3_C5_20160031_2016.01.25.17.23.13_MDA231',
                                            self.metadata, padding, shuffle=True)
        self.mda231 = torch.stack([x for x, _ in mda231_crops])

        mda468_crops = HDF5Reader.get_crops('test/data/22_384_20X-hNA_D_F_C3_C5_20160032_2016.01.25.16.27.22_MDA468',
                                            self.metadata, padding, shuffle=True)
        self.mda468 = torch.stack([x for x, _ in mda468_crops])

        if not self.avg_mda231 or not self.std_mda231:
            self.avg_mda231, self.std_mda231 = self.get_normalization_params(self.mda231)

        if not self.avg_mda468 or not self.std_mda468:
            self.avg_mda468, self.std_mda468 = self.get_normalization_params(self.mda468)

        super().__init__(self.mda231, self.mda468, self.metadata, transform)

    @staticmethod
    def read_metadata(metadata_path):
        metadata = pd.read_excel(metadata_path, engine='openpyxl')

        # Remove wells without content
        metadata = metadata[~metadata.content.isnull()]

        return metadata

    @staticmethod
    def get_normalization_params(imgs):

        imgs = torch.Tensor.float(imgs)

        avg = torch.mean(imgs, dim=0)
        std = torch.std(imgs, dim=0)

        return avg, std
