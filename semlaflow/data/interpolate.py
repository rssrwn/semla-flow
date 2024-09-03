from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

import semlaflow.util.functional as smolF
from semlaflow.util.molrepr import GeometricMol, GeometricMolBatch, SmolBatch, SmolMol

SCALE_OT_FACTOR = 0.2


_InterpT = tuple[list[SmolMol], list[SmolMol], list[SmolMol], torch.Tensor]
_GeometricInterpT = tuple[list[GeometricMol], list[GeometricMol], list[GeometricMol], torch.Tensor]


class Interpolant(ABC):
    @property
    @abstractmethod
    def hparams(self):
        pass

    @abstractmethod
    def interpolate(self, to_batch: list[SmolMol]) -> _InterpT:
        pass


class NoiseSampler(ABC):
    @property
    def hparams(self):
        pass

    @abstractmethod
    def sample_molecule(self, num_atoms: int) -> SmolMol:
        pass

    @abstractmethod
    def sample_batch(self, num_atoms: list[int]) -> SmolBatch:
        pass


class GeometricNoiseSampler(NoiseSampler):
    def __init__(
        self,
        vocab_size: int,
        n_bond_types: int,
        coord_noise: str = "gaussian",
        type_noise: str = "uniform-sample",
        bond_noise: str = "uniform-sample",
        scale_ot: bool = False,
        zero_com: bool = True,
        type_mask_index: Optional[int] = None,
        bond_mask_index: Optional[int] = None
    ):
        if coord_noise != "gaussian":
            raise NotImplementedError(f"Coord noise {coord_noise} is not supported.")

        self._check_cat_noise_type(type_noise, type_mask_index, "type")
        self._check_cat_noise_type(bond_noise, bond_mask_index, "bond")

        self.vocab_size = vocab_size
        self.n_bond_types = n_bond_types
        self.coord_noise = coord_noise
        self.type_noise = type_noise
        self.bond_noise = bond_noise
        self.scale_ot = scale_ot
        self.zero_com = zero_com
        self.type_mask_index = type_mask_index
        self.bond_mask_index = bond_mask_index

        self.coord_dist = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
        self.atomic_dirichlet = torch.distributions.Dirichlet(torch.ones(vocab_size))
        self.bond_dirichlet = torch.distributions.Dirichlet(torch.ones(n_bond_types))

    @property
    def hparams(self):
        return {
            "coord-noise": self.coord_noise,
            "type-noise": self.type_noise,
            "bond-noise": self.bond_noise,
            "noise-scale-ot": self.scale_ot,
            "zero-com": self.zero_com
        }

    def sample_molecule(self, n_atoms: int) -> GeometricMol:
        # Sample coords and scale, if required
        coords = self.coord_dist.sample((n_atoms, 3))
        if self.scale_ot:
            coords = coords * np.log(n_atoms + 1) * SCALE_OT_FACTOR

        # Sample atom types
        if self.type_noise == "dirichlet":
            atomics = self.atomic_dirichlet.sample((n_atoms,))

        elif self.type_noise == "uniform-dist":
            atomics = torch.ones((n_atoms, self.vocab_size)) / self.vocab_size

        elif self.type_noise == "mask":
            atomics = torch.zeros((n_atoms, self.vocab_size), dtype=torch.float32)
            atomics[:, self.type_mask_index] = 1.0

        elif self.type_noise == "uniform-sample":
            atomics = torch.randint(0, self.vocab_size, (n_atoms,))
            atomics = smolF.one_hot_encode_tensor(atomics, self.vocab_size)

        # Create bond indices and sample bond types
        bond_indices = torch.ones((n_atoms, n_atoms)).nonzero()
        n_bonds = bond_indices.size(0)

        if self.bond_noise == "dirichlet":
            bond_types = self.bond_dirichlet.sample((n_bonds,))

        elif self.bond_noise == "uniform-dist":
            bond_types = torch.ones((n_bonds, self.n_bond_types)) / self.n_bond_types

        elif self.bond_noise == "mask":
            bond_types = torch.tensor(self.bond_mask_index).repeat(n_bonds)
            bond_types = smolF.one_hot_encode_tensor(bond_types, self.n_bond_types)

        elif self.bond_noise == "uniform-sample":
            bond_types = torch.randint(0, self.n_bond_types, size=(n_bonds,))
            bond_types = smolF.one_hot_encode_tensor(bond_types, self.n_bond_types)

        # Create smol mol object
        mol = GeometricMol(coords, atomics, bond_indices=bond_indices, bond_types=bond_types)
        if self.zero_com:
            mol = mol.zero_com()

        return mol

    def sample_batch(self, num_atoms: list[int]) -> GeometricMolBatch:
        mols = [self.sample_molecule(n) for n in num_atoms]
        batch = GeometricMolBatch.from_list(mols)
        return batch

    def _check_cat_noise_type(self, noise_type, mask_index, name):
        if noise_type not in ["dirichlet", "uniform-dist", "mask", "uniform-sample"]:
            raise ValueError(f"{name} noise {noise_type} is not supported.")

        if noise_type == "mask" and mask_index is None:
            raise ValueError(f"{name}_mask_index must be provided if {name}_noise is 'mask'.")


class GeometricInterpolant(Interpolant):
    def __init__(
        self,
        prior_sampler: GeometricNoiseSampler,
        coord_interpolation: str = "linear",
        type_interpolation: str = "unmask",
        bond_interpolation: str = "unmask",
        coord_noise_std: float = 0.0,
        type_dist_temp: float = 1.0,
        equivariant_ot: bool = False,
        batch_ot: bool = False,
        time_alpha: float = 1.0,
        time_beta: float = 1.0,
        fixed_time: Optional[float] = None
    ):

        if fixed_time is not None and (fixed_time < 0 or fixed_time > 1):
            raise ValueError("fixed_time must be between 0 and 1 if provided.")

        if coord_interpolation != "linear":
            raise ValueError(f"coord interpolation '{coord_interpolation}' not supported.")

        if type_interpolation not in ["dirichlet", "unmask"]:
            raise ValueError(f"type interpolation '{type_interpolation}' not supported.")

        if bond_interpolation not in ["dirichlet", "unmask"]:
            raise ValueError(f"bond interpolation '{bond_interpolation}' not supported.")

        self.prior_sampler = prior_sampler
        self.coord_interpolation = coord_interpolation
        self.type_interpolation = type_interpolation
        self.bond_interpolation = bond_interpolation
        self.coord_noise_std = coord_noise_std
        self.type_dist_temp = type_dist_temp
        self.equivariant_ot = equivariant_ot
        self.batch_ot = batch_ot
        self.time_alpha = time_alpha if fixed_time is None else None
        self.time_beta = time_beta if fixed_time is None else None
        self.fixed_time = fixed_time

        self.time_dist = torch.distributions.Beta(time_alpha, time_beta)

    @property
    def hparams(self):
        prior_hparams = {f"prior-{k}": v for k, v in self.prior_sampler.hparams.items()}
        hparams = {
            "coord-interpolation": self.coord_interpolation,
            "type-interpolation": self.type_interpolation,
            "bond-interpolation": self.bond_interpolation,
            "coord-noise-std": self.coord_noise_std,
            "type-dist-temp": self.type_dist_temp,
            "equivariant-ot": self.equivariant_ot,
            "batch-ot": self.batch_ot,
            "time-alpha": self.time_alpha,
            "time-beta": self.time_beta,
            **prior_hparams
        }

        if self.fixed_time is not None:
            hparams["fixed-interpolation-time"] = self.fixed_time

        return hparams

    def interpolate(self, to_mols: list[GeometricMol]) -> _GeometricInterpT:
        batch_size = len(to_mols)
        num_atoms = max([mol.seq_length for mol in to_mols])

        from_mols = [self.prior_sampler.sample_molecule(num_atoms) for _ in to_mols]

        # Choose best possible matches for the whole batch if using batch OT
        if self.batch_ot:
            from_mols = [mol.zero_com() for mol in from_mols]
            to_mols = [mol.zero_com() for mol in to_mols]
            from_mols = self._ot_map(from_mols, to_mols)

        # Within match_mols either just truncate noise to match size of data molecule
        # Or also permute and rotate the noise to best match data molecule
        else:
            from_mols = [self._match_mols(from_mol, to_mol) for from_mol, to_mol in zip(from_mols, to_mols)]

        if self.fixed_time is not None:
            times = torch.tensor([self.fixed_time] * batch_size)
        else:
            times = self.time_dist.sample((batch_size,))

        tuples = zip(from_mols, to_mols, times.tolist())
        interp_mols = [self._interpolate_mol(from_mol, to_mol, t) for from_mol, to_mol, t in tuples]
        return from_mols, to_mols, interp_mols, list(times)

    def _ot_map(self, from_mols: list[GeometricMol], to_mols: list[GeometricMol]) -> list[GeometricMol]:
        """Permute the from_mols batch so that it forms an approximate mini-batch OT map with to_mols"""

        mol_matrix = []
        cost_matrix = []

        # Create matrix with to mols on outer axis and from mols on inner axis
        for to_mol in to_mols:
            best_from_mols = [self._match_mols(from_mol, to_mol) for from_mol in from_mols]
            best_costs = [self._match_cost(mol, to_mol) for mol in best_from_mols]
            mol_matrix.append(list(best_from_mols))
            cost_matrix.append(list(best_costs))

        row_indices, col_indices = linear_sum_assignment(np.array(cost_matrix))
        best_from_mols = [mol_matrix[r][c] for r, c in zip(row_indices, col_indices)]
        return best_from_mols

    def _match_mols(self, from_mol: GeometricMol, to_mol: GeometricMol) -> GeometricMol:
        """Permute the from_mol to best match the to_mol and return the permuted from_mol"""

        if to_mol.seq_length > from_mol.seq_length:
            raise RuntimeError("from_mol must have at least as many atoms as to_mol.")

        # Find best permutation first, then best rotation
        # As done in Equivariant Flow Matching (https://arxiv.org/abs/2306.15030)

        # Keep the same number of atoms as the data mol in the noise mol
        from_mol = from_mol.permute(list(range(to_mol.seq_length)))

        if not self.equivariant_ot:
            return from_mol

        cost_matrix = smolF.inter_distances(to_mol.coords.cpu(), from_mol.coords.cpu(), sqrd=True)
        _, from_mol_indices = linear_sum_assignment(cost_matrix.numpy())
        from_mol = from_mol.permute(from_mol_indices.tolist())

        padded_coords = smolF.pad_tensors([from_mol.coords.cpu(), to_mol.coords.cpu()])
        from_mol_coords = padded_coords[0].numpy()
        to_mol_coords = padded_coords[1].numpy()

        rotation, _ = Rotation.align_vectors(to_mol_coords, from_mol_coords)
        from_mol = from_mol.rotate(rotation)

        return from_mol

    def _match_cost(self, from_mol: GeometricMol, to_mol: GeometricMol) -> float:
        """Calculate MSE between mol coords as a match cost"""

        sqrd_dists = smolF.inter_distances(from_mol.coords.cpu(), to_mol.coords.cpu(), sqrd=True)
        mse = sqrd_dists.mean().item()
        return mse

    def _interpolate_mol(self, from_mol: GeometricMol, to_mol: GeometricMol, t: float) -> GeometricMol:
        """Interpolates mols which have already been sampled according to OT map, if required"""

        if from_mol.seq_length != to_mol.seq_length:
            raise RuntimeError("Both molecules to be interpolated must have the same number of atoms.")

        # Interpolate coords and add gaussian noise
        coords_mean = (from_mol.coords * (1 - t)) + (to_mol.coords * t)
        coords_noise = torch.randn_like(coords_mean) * self.coord_noise_std
        coords = coords_mean + coords_noise

        if self.type_interpolation == "dirichlet":
            to_atomics = torch.softmax(to_mol.atomics / self.type_dist_temp, dim=-1)
            atomics_mean = (from_mol.atomics * (1 - t)) + (to_atomics * t)
            atomics = torch.distributions.Dirichlet(atomics_mean).sample()

        elif self.type_interpolation == "unmask":
            atom_mask = torch.rand(from_mol.seq_length) > t
            to_atomics = torch.argmax(to_mol.atomics, dim=-1)
            from_atomics = torch.argmax(from_mol.atomics, dim=-1)
            to_atomics[atom_mask] = from_atomics[atom_mask]
            atomics = smolF.one_hot_encode_tensor(to_atomics, to_mol.atomics.size(-1))

        # Interpolate bonds
        if self.bond_interpolation == "dirichlet":
            to_adj = torch.softmax(to_mol.adjacency / self.type_dist_temp, dim=-1)
            adj_mean = (from_mol.adjacency * (1 - t)) + (to_adj * t)
            interp_adj = torch.distributions.Dirichlet(adj_mean).sample()

        elif self.bond_interpolation == "unmask":
            to_adj = torch.argmax(to_mol.adjacency, dim=-1)
            from_adj = torch.argmax(from_mol.adjacency, dim=-1)
            bond_mask = torch.rand_like(from_adj.float()) > t
            to_adj[bond_mask] = from_adj[bond_mask]
            interp_adj = smolF.one_hot_encode_tensor(to_adj, to_mol.adjacency.size(-1))

        bond_indices = torch.ones((from_mol.seq_length, from_mol.seq_length)).nonzero()
        bond_types = interp_adj[bond_indices[:, 0], bond_indices[:, 1]]

        interp_mol = GeometricMol(coords, atomics, bond_indices=bond_indices, bond_types=bond_types)
        return interp_mol
