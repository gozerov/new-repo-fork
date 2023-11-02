import os

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from torchvision.datasets import CIFAR10

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = PATH_DATASETS):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.dims = (1, 28, 28)
        self.num_classes = 10
        self.classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar10_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=BATCH_SIZE)