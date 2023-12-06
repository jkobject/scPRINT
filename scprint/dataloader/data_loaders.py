from torch.utils.data.dataloader import default_collate
from scprint.base import BaseDataLoader


class DataLoader(BaseDataLoader):
    """
    Base class for all data loaders
    """

    def __init__(
        self,
    ):
        pass

    def split_validation(self):
        pass
