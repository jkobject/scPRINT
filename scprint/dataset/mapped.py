from collections import Counter
from os import PathLike
from typing import List, Optional, Union

import numpy as np
from lamindb_setup.dev.upath import UPath

from lamindb.dev.storage._backed_access import (
    ArrayTypes,
    GroupTypes,
    StorageType,
    registry,
)

from typing import Iterable, List, Optional, Union

from lamin_utils import logger
from lnschema_core.models import (
    Data,
    Run,
    __repr__,
)
from lamindb.dev._settings import settings
from lamindb.dev._run_context import run_context
from lamindb.dev._data import _track_run_input


def mapped(
    dataset,
    label_keys: Optional[Union[str, List[str]]] = None,
    encode_labels: Optional[Union[bool, List[str]]] = False,
    stream: bool = False,
    is_run_input: Optional[bool] = None,
) -> "MappedDataset":
    _track_run_input(dataset, is_run_input)
    path_list = []
    for file in dataset.files.all():
        if file.suffix not in {".h5ad", ".zrad", ".zarr"}:
            logger.warning(f"Ignoring file with suffix {file.suffix}")
            continue
        elif not stream and file.suffix == ".h5ad":
            path_list.append(file.stage())
        else:
            path_list.append(file.path)
    return MappedDataset(path_list, label_keys, encode_labels)


class MappedDataset:
    """Map-style dataset for use in data loaders.

    This currently only works for collections of `AnnData` objects.

    For an example, see :meth:`~lamindb.Dataset.mapped`.

    .. note::

        A similar data loader exists `here
        <https://github.com/Genentech/scimilarity>`__.
    """

    def __init__(
        self,
        path_list: List[Union[str, PathLike]],
        label_keys: Optional[Union[str, List[str]]] = None,
        encode_labels: Optional[Union[bool, List[str]]] = False,
    ):
        self.storages = []
        self.conns = []
        for path in path_list:
            path = UPath(path)
            if path.exists() and path.is_file():  # type: ignore
                conn, storage = registry.open("h5py", path)
            else:
                conn, storage = registry.open("zarr", path)
            self.conns.append(conn)
            self.storages.append(storage)

        self.n_obs_list = []
        for storage in self.storages:
            X = storage["X"]
            if isinstance(X, ArrayTypes):  # type: ignore
                self.n_obs_list.append(X.shape[0])
            else:
                self.n_obs_list.append(X.attrs["shape"][0])
        self.n_obs = sum(self.n_obs_list)

        self.indices = np.hstack([np.arange(n_obs) for n_obs in self.n_obs_list])
        self.storage_idx = np.repeat(np.arange(len(self.storages)), self.n_obs_list)

        if isinstance(label_keys, str):
            label_keys = [label_keys]
        if isinstance(encode_labels, bool):
            if encode_labels:
                encode_labels = label_keys
            else:
                encode_labels = []
        if isinstance(encode_labels, list):
            self.encoders = {}
            for label in encode_labels:
                cats = self.get_merged_categories(label)
                self.encoders[label] = {cat: i for i, cat in enumerate(cats)}
        else:
            self.encoders = {}
        self.label_keys = label_keys
        self._closed = False

    def __len__(self):
        return self.n_obs

    def __getitem__(self, idx):
        obs_idx = self.indices[idx]
        labels = []
        if isinstance(idx, slice):
            out = []
            label = []
            for i in range(idx.start, idx.stop):
                pout, plabel = self.__getitem__(i)
                out.append(pout)
                label.append(plabel)
            return np.array(out), label
        storage = self.storages[self.storage_idx[idx]]
        out = self.get_data_idx(storage, obs_idx)
        if self.label_keys is not None:
            for label in self.label_keys:
                label_idx = self.get_label_idx(storage, obs_idx, label)
                if label in self.encoders:
                    labels.append(self.encoders[label][label_idx])
                else:
                    labels.append(label_idx)
        return out, labels

    def uns(self, idx, key):
        storage = self.storages[self.storage_idx[idx]]
        return storage["uns"][key]

    def get_data_idx(
        self, storage: StorageType, idx, layer_key: Optional[str] = None  # type: ignore # noqa
    ):
        """Get the index for the data."""
        layer = storage["X"] if layer_key is None else storage["layers"][layer_key]  # type: ignore # noqa
        if isinstance(layer, ArrayTypes):  # type: ignore
            return layer[idx]
        else:  # assume csr_matrix here
            data = layer["data"]
            indices = layer["indices"]
            indptr = layer["indptr"]
            if isinstance(idx, slice):
                s = [slice(*(indptr[i : i + 2])) for i in range(idx.start, idx.stop)]
                layer_idx = np.zeros((layer.attrs["shape"][1], len(s)))
                for i, sl in enumerate(s):
                    layer_idx[indices[sl], i] = data[sl]
            else:  # assuming int
                s = slice(*(indptr[idx : idx + 2]))
                layer_idx = np.zeros(layer.attrs["shape"][1])
                layer_idx[indices[s]] = data[s]
            return layer_idx

    def get_label_idx(self, storage: StorageType, idx: int, label_key: str):  # type: ignore # noqa
        """Get the index for the label by key."""
        obs = storage["obs"]  # type: ignore
        # how backwards compatible do we want to be here actually?
        if isinstance(obs, ArrayTypes):  # type: ignore
            label = obs[idx][obs.dtype.names.index(label_key)]
        else:
            labels = obs[label_key]
            if isinstance(labels, ArrayTypes):  # type: ignore
                label = labels[idx]
            else:
                label = labels["codes"][idx]

        cats = self.get_categories(storage, label_key)
        if cats is not None:
            label = cats[label]
        if isinstance(label, bytes):
            label = label.decode("utf-8")
        return label

    def get_label_weights(self, label_keys: Union[str, List[str]], scaler=10):
        """Get all weights for a given label key."""
        if type(label_keys) is not list:
            label_keys = [label_keys]
        for i, val in enumerate(label_keys):
            if val not in self.label_keys:
                raise ValueError(f"{val} is not a valid label key.")
            if i == 0:
                labels = self.get_merged_labels(val)
            else:
                labels += "_" + self.get_merged_labels(val)
        counter = Counter(labels)  # type: ignore
        counter = np.array([counter[label] for label in labels])
        weights = scaler / (counter + scaler)
        return weights

    def get_merged_labels(self, label_key: str):
        """Get merged labels."""
        labels_merge = []
        decode = np.frompyfunc(lambda x: x.decode("utf-8"), 1, 1)
        for storage in self.storages:
            codes = self.get_codes(storage, label_key)
            labels = decode(codes) if isinstance(codes[0], bytes) else codes
            cats = self.get_categories(storage, label_key)
            if cats is not None:
                cats = decode(cats) if isinstance(cats[0], bytes) else cats
                labels = cats[labels]
            labels_merge.append(labels)
        return np.hstack(labels_merge)

    def get_merged_categories(self, label_key: str):
        """Get merged categories."""
        cats_merge = set()
        decode = np.frompyfunc(lambda x: x.decode("utf-8"), 1, 1)
        for storage in self.storages:
            cats = self.get_categories(storage, label_key)
            if cats is not None:
                cats = decode(cats) if isinstance(cats[0], bytes) else cats
                cats_merge.update(cats)
            else:
                codes = self.get_codes(storage, label_key)
                codes = decode(codes) if isinstance(codes[0], bytes) else codes
                cats_merge.update(codes)
        return cats_merge

    def get_categories(self, storage: StorageType, label_key: str):  # type: ignore
        """Get categories."""
        obs = storage["obs"]  # type: ignore
        if isinstance(obs, ArrayTypes):  # type: ignore
            cat_key_uns = f"{label_key}_categories"
            if cat_key_uns in storage["uns"]:  # type: ignore
                return storage["uns"][cat_key_uns]  # type: ignore
            else:
                return None
        else:
            if "__categories" in obs:
                cats = obs["__categories"]
                if label_key in cats:
                    return cats[label_key]
                else:
                    return None
            labels = obs[label_key]
            if isinstance(labels, GroupTypes):  # type: ignore
                if "categories" in labels:
                    return labels["categories"]
                else:
                    return None
            else:
                if "categories" in labels.attrs:
                    return labels.attrs["categories"]
                else:
                    return None

    def get_codes(self, storage: StorageType, label_key: str):  # type: ignore
        """Get codes."""
        obs = storage["obs"]  # type: ignore
        if isinstance(obs, ArrayTypes):  # type: ignore
            label = obs[label_key]
        else:
            label = obs[label_key]
            if isinstance(label, ArrayTypes):  # type: ignore
                return label[...]
            else:
                return label["codes"][...]

    def close(self):
        """Close connection to array streaming backend."""
        for storage in self.storages:
            if hasattr(storage, "close"):
                storage.close()
        for conn in self.conns:
            if hasattr(conn, "close"):
                conn.close()
        self._closed = True

    @property
    def closed(self):
        return self._closed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
