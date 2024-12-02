import threading
from typing import Optional, Union

import numpy as np
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation

ArrT = np.ndarray


# *************************************************************************************************
# ************************************ Periodic Table class ***************************************
# *************************************************************************************************


class PeriodicTable:
    """Singleton class wrapper for the RDKit periodic table providing a neater interface"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self):
        self._table = Chem.GetPeriodicTable()

        # Just to be certain that vocab objects are thread safe
        self._pt_lock = threading.Lock()

    def atomic_from_symbol(self, symbol: str) -> int:
        with self._pt_lock:
            symbol = symbol.upper() if len(symbol) == 1 else symbol
            atomic = self._table.GetAtomicNumber(symbol)

        return atomic

    def symbol_from_atomic(self, atomic_num: int) -> str:
        with self._pt_lock:
            token = self._table.GetElementSymbol(atomic_num)

        return token

    def valence(self, atom: Union[str, int]) -> int:
        with self._pt_lock:
            valence = self._table.GetDefaultValence(atom)

        return valence


# *************************************************************************************************
# ************************************* Global Declarations ***************************************
# *************************************************************************************************


PT = PeriodicTable()

IDX_BOND_MAP = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
BOND_IDX_MAP = {bond: idx for idx, bond in IDX_BOND_MAP.items()}

IDX_CHARGE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: -1, 5: -2, 6: -3}
CHARGE_IDX_MAP = {charge: idx for idx, charge in IDX_CHARGE_MAP.items()}


# *************************************************************************************************
# *************************************** Util Functions ******************************************
# *************************************************************************************************

# TODO merge these with check functions in other files


def _check_shape_len(arr, allowed, name="object"):
    num_dims = len(arr.shape)
    allowed = [allowed] if isinstance(allowed, int) else allowed
    if num_dims not in allowed:
        raise RuntimeError(f"Number of dimensions of {name} must be in {str(allowed)}, got {num_dims}")


def _check_dim_shape(arr, dim, allowed, name="object"):
    shape = arr.shape[dim]
    allowed = [allowed] if isinstance(allowed, int) else allowed
    if shape not in allowed:
        raise RuntimeError(f"Shape of {name} for dim {dim} must be in {allowed}, got {shape}")


# *************************************************************************************************
# ************************************* External Functions ****************************************
# *************************************************************************************************


def mol_is_valid(mol: Chem.rdchem.Mol, with_hs: bool = True, connected: bool = True) -> bool:
    """Whether the mol can be sanitised and, optionally, whether it's fully connected

    Args:
        mol (Chem.Mol): RDKit molecule to check
        with_hs (bool): Whether to check validity including hydrogens (if they are in the input mol), default True
        connected (bool): Whether to also assert that the mol must not have disconnected atoms, default True

    Returns:
        bool: Whether the mol is valid
    """

    if mol is None:
        return False

    mol_copy = Chem.Mol(mol)
    if not with_hs:
        mol_copy = Chem.RemoveAllHs(mol_copy)

    try:
        AllChem.SanitizeMol(mol_copy)
    except Exception:
        return False

    n_frags = len(AllChem.GetMolFrags(mol_copy))
    if connected and n_frags != 1:
        return False

    return True


def calc_energy(mol: Chem.rdchem.Mol, per_atom: bool = False) -> float:
    """Calculate the energy for an RDKit molecule using the MMFF forcefield

    The energy is only calculated for the first (0th index) conformer within the molecule. The molecule is copied so
    the original is not modified.

    Args:
        mol (Chem.Mol): RDKit molecule
        per_atom (bool): Whether to normalise by number of atoms in mol, default False

    Returns:
        float: Energy of the molecule or None if the energy could not be calculated
    """

    mol_copy = Chem.Mol(mol)

    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_copy, mmffVariant="MMFF94")
        ff = AllChem.MMFFGetMoleculeForceField(mol_copy, mmff_props, confId=0)
        energy = ff.CalcEnergy()
        energy = energy / mol.GetNumAtoms() if per_atom else energy
    except Exception:
        energy = None

    return energy


def optimise_mol(mol: Chem.rdchem.Mol, max_iters: int = 1000) -> Chem.rdchem.Mol:
    """Optimise the conformation of an RDKit molecule

    Only the first (0th index) conformer within the molecule is optimised. The molecule is copied so the original
    is not modified.

    Args:
        mol (Chem.Mol): RDKit molecule
        max_iters (int): Max iterations for the conformer optimisation algorithm

    Returns:
        Chem.Mol: Optimised molecule or None if the molecule could not be optimised within the given number of
                iterations
    """

    mol_copy = Chem.Mol(mol)
    try:
        AllChem.MMFFOptimizeMolecule(mol_copy, maxIters=max_iters)
    except Exception:
        return None

    return mol_copy


def conf_distance(mol1: Chem.rdchem.Mol, mol2: Chem.rdchem.Mol, fix_order: bool = True) -> float:
    """Approximately align two molecules and then calculate RMSD between them

    Alignment and distance is calculated only between the default conformers of each molecule.

    Args:
        mol1 (Chem.Mol): First molecule to align
        mol2 (Chem.Mol): Second molecule to align
        fix_order (bool): Whether to fix the atom order of the molecules

    Returns:
        float: RMSD between molecules after approximate alignment
    """

    assert len(mol1.GetAtoms()) == len(mol2.GetAtoms())

    if not fix_order:
        raise NotImplementedError()

    coords1 = np.array(mol1.GetConformer().GetPositions())
    coords2 = np.array(mol2.GetConformer().GetPositions())

    # Firstly, centre both molecules
    coords1 = coords1 - (coords1.sum(axis=0) / coords1.shape[0])
    coords2 = coords2 - (coords2.sum(axis=0) / coords2.shape[0])

    # Find the best rotation alignment between the centred mols
    rotation, _ = Rotation.align_vectors(coords1, coords2)
    aligned_coords2 = rotation.apply(coords2)

    sqrd_dists = (coords1 - aligned_coords2) ** 2
    rmsd = np.sqrt(sqrd_dists.sum(axis=1).mean())
    return rmsd


# TODO could allow more args
def smiles_from_mol(mol: Chem.rdchem.Mol, canonical: bool = True, explicit_hs: bool = False) -> Union[str, None]:
    """Create a SMILES string from a molecule

    Args:
        mol (Chem.Mol): RDKit molecule object
        canonical (bool): Whether to create a canonical SMILES, default True
        explicit_hs (bool): Whether to embed hydrogens in the mol before creating a SMILES, default False. If True
                this will create a new mol with all hydrogens embedded. Note that the SMILES created by doing this
                is not necessarily the same as creating a SMILES showing implicit hydrogens.

    Returns:
        str: SMILES string which could be None if the SMILES generation failed
    """

    if mol is None:
        return None

    if explicit_hs:
        mol = Chem.AddHs(mol)

    try:
        smiles = Chem.MolToSmiles(mol, canonical=canonical)
    except Exception:
        smiles = None

    return smiles


def mol_from_smiles(smiles: str, explicit_hs: bool = False) -> Union[Chem.rdchem.Mol, None]:
    """Create a RDKit molecule from a SMILES string

    Args:
        smiles (str): SMILES string
        explicit_hs (bool): Whether to embed explicit hydrogens into the mol

    Returns:
        Chem.Mol: RDKit molecule object or None if one cannot be created from the SMILES
    """

    if smiles is None:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol) if explicit_hs else mol
    except Exception:
        mol = None

    return mol


def mol_from_atoms(
    coords: ArrT, tokens: list[str], bonds: Optional[ArrT] = None, charges: Optional[ArrT] = None, sanitise=True
):
    """Create RDKit mol from atom coords and atom tokens (and optionally bonds)

    If any of the atom tokens are not valid atoms (do not exist on the periodic table), None will be returned.

    If bonds are not provided this function will create a partial molecule using the atomics and coordinates and then
    infer the bonds based on the coordinates using OpenBabel. Otherwise the bonds are added to the molecule as they
    are given in the bond array.

    If bonds are provided they must not contain any duplicates.

    If charges are not provided they are assumed to be 0 for all atoms.

    Args:
        coords (np.ndarray): Coordinate tensor, shape [n_atoms, 3]
        atomics (list[str]): Atomic numbers, length must be n_atoms
        bonds (np.ndarray, optional): Bond indices and types, shape [n_bonds, 3]
        charges (np.ndarray, optional): Charge for each atom, shape [n_atoms]
        sanitise (bool): Whether to apply RDKit sanitization to the molecule, default True

    Returns:
        Chem.rdchem.Mol: RDKit molecule or None if one cannot be created
    """

    _check_shape_len(coords, 2, "coords")
    _check_dim_shape(coords, 1, 3, "coords")

    if coords.shape[0] != len(tokens):
        raise ValueError("coords and atomics tensor must have the same number of atoms.")

    if bonds is not None:
        _check_shape_len(bonds, 2, "bonds")
        _check_dim_shape(bonds, 1, 3, "bonds")

    if charges is not None:
        _check_shape_len(charges, 1, "charges")
        _check_dim_shape(charges, 0, len(tokens), "charges")

    try:
        atomics = [PT.atomic_from_symbol(token) for token in tokens]
    except Exception:
        return None

    charges = charges.tolist() if charges is not None else [0] * len(tokens)

    # Add atom types and charges
    mol = Chem.EditableMol(Chem.Mol())
    for idx, atomic in enumerate(atomics):
        atom = Chem.Atom(atomic)
        atom.SetFormalCharge(charges[idx])
        mol.AddAtom(atom)

    # Add 3D coords
    conf = Chem.Conformer(coords.shape[0])
    for idx, coord in enumerate(coords.tolist()):
        conf.SetAtomPosition(idx, coord)

    mol = mol.GetMol()
    mol.AddConformer(conf)

    if bonds is None:
        return _infer_bonds(mol)

    # Add bonds if they have been provided
    mol = Chem.EditableMol(mol)
    for bond in bonds.astype(np.int32).tolist():
        start, end, b_type = bond

        if b_type not in IDX_BOND_MAP:
            return None

        # Don't add self connections
        if start != end:
            b_type = IDX_BOND_MAP[b_type]
            mol.AddBond(start, end, b_type)

    try:
        mol = mol.GetMol()
        for atom in mol.GetAtoms():
            atom.UpdatePropertyCache(strict=False)
    except Exception:
        return None

    if sanitise:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

    return mol


def _infer_bonds(mol: Chem.rdchem.Mol):
    coords = mol.GetConformer().GetPositions().tolist()
    coord_strs = ["\t".join([f"{c:.6f}" for c in cs]) for cs in coords]
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    xyz_str_header = f"{str(mol.GetNumAtoms())}\n\n"
    xyz_strs = [f"{str(atom)}\t{coord_str}" for coord_str, atom in zip(coord_strs, atom_symbols)]
    xyz_str = xyz_str_header + "\n".join(xyz_strs)

    try:
        pybel_mol = pybel.readstring("xyz", xyz_str)
    except Exception:
        pybel_mol = None

    if pybel_mol is None:
        return None

    mol_str = pybel_mol.write("mol")
    mol = Chem.MolFromMolBlock(mol_str, removeHs=False, sanitize=True)
    return mol
