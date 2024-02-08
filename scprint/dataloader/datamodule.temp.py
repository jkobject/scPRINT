# import datamodule
import pytorch_lightning as pl
from torchvision import transforms
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from scprint.dataset import Dataset
import lamindb as ln
from torch.utils.data import DataLoader
from typing import Optional


class DatasetModule(pl.LightningDataModule):
    def __init__(
        self,
        data_name: str,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ):
        super().__init__()
        self.dataset = Dataset(ln.Collection.filter(name="preprocessed dataset").one())
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.validation_split = validation_split
        self.train_sampler, self.valid_sampler = self._split_sampler(
            self.validation_split
        )

    def _split_sampler(self, split, weighting: int = 30):
        idx_full = np.arange(self.n_samples)
        np.random.shuffle(idx_full)
        weights = self.dataset.get_label_weights(weighting)
        if isinstance(split, int):
            assert (
                split < self.n_samples
            ), "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)
        if len_valid == 0:
            self.train_idx = idx_full
        else:
            self.valid_idx = idx_full[0:len_valid]
            self.train_idx = np.delete(idx_full, np.arange(0, len_valid))
            valid_weights = weights[self.valid_idx]
            # TODO: should we do weighted random sampling for validation set?
            valid_sampler = WeightedRandomSampler(valid_weights, len_valid)
        train_weights = weights[self.train_idx]
        train_sampler = WeightedRandomSampler(train_weights, len(self.train_idx))
        # turn off shuffle option which is mutually exclusive with sampler

        return train_sampler, valid_sampler if len_valid != 0 else train_sampler, None

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def setup(self, stage: Optional[str] = None):
        return True

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        return True
