from typing import List, Optional, Union
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset as BaseDataset
from sklearn import datasets
import torch
from config import BATCH_SIZE


class Dataset(LightningDataModule):
    def __init__(
        self,
        train_batch_size=BATCH_SIZE,
        val_batch_size=BATCH_SIZE,
        num_workers=8,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = MnistDataset(0, 1200)
        self.val_dataset = MnistDataset(1201, 1792)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class MnistDataset(BaseDataset):
    def __init__(self, start_index, end_index):
        self.digits = datasets.load_digits().images[start_index:end_index]

    def __len__(self):
        return len(self.digits)

    def __getitem__(self, item):
        return torch.Tensor(self.digits[item] / 16).float()
