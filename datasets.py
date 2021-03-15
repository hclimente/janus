import random

from os.path import join
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

        # give all moas the same chance
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
        return 100000

    @staticmethod
    def sample_crops(prev_moa, same_moa, dataset):
        while True:
            crop, info = random.choice(dataset)

            if same_moa and prev_moa == info['moa']:
                break
            elif not same_moa and prev_moa != info['moa']:
                break

        return crop, info['moa']


class Boyd2019(MultiCellDataset):

    def __init__(self, data_path, metadata,
                 transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.RandomVerticalFlip(),
                                               RandomRot90()])):

        padding = 32

        mda231_crops = HDF5Reader.get_crops(join(data_path,
                                                 '22_384_20X-hNA_D_F_C3_C5_20160031_2016.01.25.17.23.13_MDA231'),
                                            metadata, padding)
        mda231 = [x for x in mda231_crops]
        self.avg_mda231, self.std_mda231 = self.load_parameters(join(data_path, 'mda231_params.pkl'),
                                                                torch.stack([x[0] for x in mda231]))

        mda468_crops = HDF5Reader.get_crops(join(data_path,
                                                 '22_384_20X-hNA_D_F_C3_C5_20160032_2016.01.25.16.27.22_MDA468'),
                                            metadata, padding)
        mda468 = [x for x in mda468_crops]
        self.avg_mda468, self.std_mda468 = self.load_parameters(join(data_path, 'mda468_params.pkl'),
                                                                torch.stack([x[0] for x in mda468]))

        super().__init__(mda231, mda468, metadata, transform)

    @staticmethod
    def read_metadata(metadata_path):
        metadata = pd.read_excel(metadata_path, engine='openpyxl')

        # Remove empty wells
        metadata = metadata[~metadata.content.isnull()]

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
                avg, std = pickle.load(parameters)
        except IOError:
            with open(pickle_path, 'wb') as parameters:
                avg, std = Boyd2019.get_normalization_params(crops)
                pickle.dump((avg, std), parameters)

        return avg, std
