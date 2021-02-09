import random

from torchvision import transforms
from torch.utils.data import Dataset

from transforms import RandomVerticalFlip,RandomHorizontalFlip,RandomRotation


class MultiCellDataset (Dataset):

    def __init__(self, dataset_1, dataset_2,
                 transform=transforms.Compose([RandomHorizontalFlip(),
                                               RandomVerticalFlip(),
                                               RandomRotation()])):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.transform = transform

    def __getitem__(self, index):
        random.seed(index)
        cell1, moa1 = random.choice(self.dataset_1)

        same_moa = random.getrandbits(1)

        while True:
            cell2, moa2 = random.choice(self.dataset_2)

            if same_moa and moa1 == moa2:
                break
            elif not same_moa and moa1 != moa2:
                break

        if self.transform:
            cell1 = self.transform(cell1)
            cell2 = self.transform(cell2)

        return cell1, cell2, same_moa

    def __len__(self):
        return int(len(self.dataset_1) * len(self.dataset_2) / 2)


class Boyd2019(MultiCellDataset):

    def __init__(self, path_mda231, path_mda468, transform):

        window_width = 32

        dataset_1 = self.load_dataset(path_mda231)
        dataset_2 = self.load_dataset(path_mda468)

        super().__init__(dataset_1, dataset_2, transform)

    def load_dataset(self, path):
        return path
