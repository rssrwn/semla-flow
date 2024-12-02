from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from semlaflow.util.molrepr import GeometricMolBatch

# *** Util functions ***


def load_smol_data(data_path, smol_cls):
    data_path = Path(data_path)

    # TODO handle having a directory with batched data files
    if data_path.is_dir():
        raise NotImplementedError()

    # TODO maybe read in chunks if this is too big
    bytes_data = data_path.read_bytes()
    data = smol_cls.from_bytes(bytes_data)
    return data


# *** Abstract class for all Smol data types ***


class SmolDataset(ABC, torch.utils.data.Dataset):
    def __init__(self, smol_data, transform=None):
        super().__init__()

        self._data = smol_data
        self.transform = transform

    @property
    def hparams(self):
        return {}

    @property
    def lengths(self):
        return self._data.seq_length

    def __len__(self):
        return self._data.batch_size

    def __getitem__(self, item):
        molecule = self._data[item]
        if self.transform is not None:
            molecule = self.transform(molecule)

        return molecule

    @classmethod
    @abstractmethod
    def load(cls, data_path, transform=None):
        pass


# *** SmolDataset implementations ***


class GeometricDataset(SmolDataset):
    def sample(self, n_items, replacement=False):
        mol_samples = np.random.choice(self._data.to_list(), n_items, replace=replacement)
        data = GeometricMolBatch.from_list(mol_samples)
        return GeometricDataset(data, transform=self.transform)

    @classmethod
    def load(cls, data_path, transform=None, min_size=None):
        data = load_smol_data(data_path, GeometricMolBatch)
        if min_size is not None:
            mols = [mol for mol in data if mol.seq_length >= min_size]
            data = GeometricMolBatch.from_list(mols)

        return GeometricDataset(data, transform=transform)


# *** Other useful datasets ***


class SmolPairDataset(torch.utils.data.Dataset):
    """A dataset which returns pairs of SmolMol objects"""

    def __init__(self, from_dataset: SmolDataset, to_dataset: SmolDataset):
        super().__init__()

        if len(from_dataset) != len(to_dataset):
            raise ValueError("From and to datasets must have the same number of items.")

        if from_dataset.lengths != to_dataset.lengths:
            raise ValueError("From and to datasets must have molecules of the same length at each index.")

        self.from_dataset = from_dataset
        self.to_dataset = to_dataset

    # TODO stop hparams clashing from different sources
    @property
    def hparams(self):
        return {**self.from_dataset.hparams, **self.to_dataset.hparams}

    @property
    def lengths(self):
        return self.from_dataset.lengths

    def __len__(self):
        return len(self.from_dataset)

    def __getitem__(self, item):
        from_mol = self.from_dataset[item]
        to_mol = self.to_dataset[item]
        return from_mol, to_mol
