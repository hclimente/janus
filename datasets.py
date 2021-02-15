import random

from torchvision import transforms
from torch.utils.data import Dataset


class MultiCellDataset (Dataset):

    def __init__(self, dataset_1, dataset_2,
                 transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.RandomVerticalFlip(),
                                               transforms.RandomRotation(90)])):

        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.moas = list(set([x[1] for x in dataset_1] + [x[1] for x in dataset_2]))
        self.transform = transform

    def __getitem__(self, index, balanced=True):
        random.seed(index)

        if balanced:
            moa1 = random.choice(self.moas)
            cell1, moa1 = self.__sample_cells(moa1, True)
        else:
            cell1, moa1 = random.choice(self.dataset_1)

        same_moa = random.getrandbits(1)
        cell2, moa2 = self.__sample_cells(moa1, same_moa)

        if self.transform:
            cell1 = self.transform(cell1)
            cell2 = self.transform(cell2)

        return cell1, moa1, cell2, moa2, same_moa

    def __len__(self):
        # artificially limit epoch length
        return 10000

    def __sample_cells(self, moa1, same_moa):
        while True:
            cell2, moa2 = random.choice(self.dataset_2)

            if same_moa and moa1 == moa2:
                break
            elif not same_moa and moa1 != moa2:
                break

        return cell2, moa2


class Boyd2019(MultiCellDataset):

    def __init__(self, path_mda231, path_mda468, transform):

        window_width = 32

        dataset_1 = self.load_dataset(path_mda231)
        dataset_2 = self.load_dataset(path_mda468)

        super().__init__(dataset_1, dataset_2, transform)

    def load_dataset(self, path):
        return path
