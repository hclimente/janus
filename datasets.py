from os.path import join
import random

import numpy as np
import pandas as pd
import pickle
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

    def __getitem__(self, index):
        random.seed(index)

        # give all moas the same weight
        moa1 = random.choice(self.metadata['moa'].unique())
        cell1, moa1 = self.sample_crops(moa1, True, self.dataset_1)

        same_moa = random.getrandbits(1)
        cell2, moa2 = self.sample_crops(moa1, same_moa, self.dataset_2)

        if self.transform:
            cell1 = self.transform(cell1)
            cell2 = self.transform(cell2)

        return cell1, moa1, cell2, moa2, same_moa

    def __len__(self):
        # artificially limit epoch length
        return 10000

    @staticmethod
    def sample_crops(prev_moa, same_moa, dataset):
        selection = np.array([x['moa'] == prev_moa for _, x in dataset])

        if not same_moa:
            selection = ~selection

        list_idx = np.where(selection)[0]
        idx = np.random.choice(list_idx)

        return dataset[idx][0], dataset[idx][1]['moa']


class Boyd2019(MultiCellDataset):

    def __init__(self, data_path, metadata, padding = 32, force_calc_params = True,
                 transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.RandomVerticalFlip(),
                                               RandomRot90()])):

        mda231 = self.load_crops(join(data_path, '22_384_20X-hNA_D_F_C3_C5_20160031_2016.01.25.17.23.13_MDA231'),
                                 metadata, padding, force_calc_params)
        mda468 = self.load_crops(join(data_path, '22_384_20X-hNA_D_F_C3_C5_20160032_2016.01.25.16.27.22_MDA468'),
                                 metadata, padding, force_calc_params)

        super().__init__(mda231, mda468, metadata, transform)

    @staticmethod
    def read_metadata(metadata_path):
        metadata = pd.read_excel(metadata_path, engine='openpyxl')

        # remove empty wells
        metadata = metadata[~metadata.content.isnull()]

        # remove wells with no drug/dmso
        metadata = metadata[metadata['content'] != 'None']

        return metadata

    @staticmethod
    def get_normalization_params(imgs):

        imgs = torch.Tensor.float(imgs)

        avg = torch.mean(imgs, dim=0)
        std = torch.std(imgs, dim=0)

        return avg, std

    @staticmethod
    def load_parameters(pickle_path, crops):

        try:
            with open(pickle_path, 'rb') as parameters:
                print('loading normalization parameters')
                avg, std = pickle.load(parameters)
        except IOError:
            print('computing normalization parameters')
            with open(pickle_path, 'wb') as parameters:
                avg, std = Boyd2019.get_normalization_params(crops)
                pickle.dump((avg, std), parameters)

        return avg, std

    @staticmethod
    def load_crops(crops_path, metadata, padding, normalize=True):

        crops = HDF5Reader.get_crops(crops_path, metadata, padding)
        crops = [x for x in crops]
        if normalize:
            avg, std = Boyd2019.load_parameters(join(crops_path, 'norm_params.pkl'),
                                                torch.stack([x[0] for x in crops]))
            crops = [((x[0] - avg)/std, x[1]) for x in crops]

        return crops
