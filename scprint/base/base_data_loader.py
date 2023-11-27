import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

from anndata.experimental import AnnLoader

# TODO: put in config
COARSE_TISSUE = {
    "adipose tissue": "",
    "bladder organ": "",
    "blood": "",
    "bone marrow": "",
    "brain": "",
    "breast": "",
    "esophagus": "",
    "eye": "",
    "embryo": "",
    "fallopian tube": "",
    "gall bladder": "",
    "heart": "",
    "intestine": "",
    "kidney": "",
    "liver": "",
    "lung": "",
    "lymph node": "",
    "musculature of body": "",
    "nose": "",
    "ovary": "",
    "pancreas": "",
    "placenta": "",
    "skin of body": "",
    "spinal cord": "",
    "spleen": "",
    "stomach": "",
    "thymus": "",
    "thyroid gland": "",
    "tongue": "",
    "uterus": "",
}

COARSE_ANCESTRY = {
    "African": "",
    "Chinese": "",
    "East Asian": "",
    "Eskimo": "",
    "European": "",
    "Greater Middle Eastern  (Middle Eastern, North African or Persian)": "",
    "Hispanic or Latin American": "",
    "Native American": "",
    "Oceanian": "",
    "South Asian": "",
}

COARSE_DEVELOPMENT_STAGE = {
    "Embryonic human": "",
    "Fetal": "",
    "Immature": "",
    "Mature": "",
}

COARSE_ASSAY = {
    "10x 3'": "",
    "10x 5'": "",
    "10x multiome": "",
    "CEL-seq2": "",
    "Drop-seq": "",
    "GEXSCOPE technology": "",
    "inDrop": "",
    "microwell-seq": "",
    "sci-Plex": "",
    "sci-RNA-seq": "",
    "Seq-Well": "",
    "Slide-seq": "",
    "Smart-seq": "",
    "SPLiT-seq": "",
    "TruDrop": "",
    "Visium Spatial Gene Expression": "",
}
CELL_ONTO: str = "https://github.com/obophenotype/cell-ontology/releases/latest/download/cl-basic.owl"
TISSUE_ONTO: str = "https://github.com/obophenotype/uberon/releases/latest/download/uberon-basic.owl"
ANCESTRY_ONTO: str = "https://raw.githubusercontent.com/EBISPOT/hancestro/main/hancestro-base.owl"
ASSAY_ONTO: str = "https://github.com/obophenotype/uberon/releases/latest/download/uberon-basic.owl"
DEVELOPMENT_STAGE_ONTO: str = "http://purl.obolibrary.org/obo/hsapdv.owl"
DISEASE_ONTO: str = 'https://raw.githubusercontent.com/EBISPOT/efo/master/efo-base.owl'


class BaseDataLoader(AnnLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, anndataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle
        
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)




def weighted_random_mask_value(
    values: Union[torch.Tensor, np.ndarray],
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    important_elements: Union[torch.Tensor, np.ndarray]=np.array([]),
    important_weight: int = 0,
    pad_value: int = 0,
) -> torch.Tensor:
    """
    Randomly mask a batch of data.

    Args:
        values (array-like):
            A batch of tokenized data, with shape (batch_size, n_features).
        mask_ratio (float): The ratio of genes to mask, default to 0.15.
        mask_value (int): The value to mask with, default to -1.
        pad_value (int): The value of padding in the values, will be kept unchanged.

    Returns:
        torch.Tensor: A tensor of masked data.
    """
    if isinstance(values, torch.Tensor):
        # it is crutial to clone the tensor, otherwise it changes the original tensor
        values = values.clone().detach().numpy()
    else:
        values = values.copy()

    for i in range(len(values)):
        row = values[i]
        non_padding_idx = np.nonzero(row - pad_value)[0]
        non_padding_idx = np.setdiff1d(non_padding_idx, do_not_pad_index)
        n_mask = int(len(non_padding_idx) * mask_ratio)
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
        row[mask_idx] = mask_value
    return torch.from_numpy(values).float()