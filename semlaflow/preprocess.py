"""Preprocessing script only for Geom Drugs, QM9 is done in the QM9 notebook"""

import pickle
import argparse
from pathlib import Path

from semlaflow.util.molrepr import GeometricMol, GeometricMolBatch


DEFAULT_RAW_DATA_FOLDER = "raw"
DEFUALT_SAVE_DATA_FOLDER = "smol"


RAW_TRAIN_FILE = "train_data.pickle"
RAW_VAL_FILE = "val_data.pickle"
RAW_TEST_FILE = "test_data.pickle"

SAVE_TRAIN_FILE = "train.smol"
SAVE_VAL_FILE = "val.smol"
SAVE_TEST_FILE = "test.smol"


def read_from_file(filepath):
    bytes = filepath.read_bytes()
    return pickle.loads(bytes)


def raw_to_smol_mol(raw_mol):
    _, mols = raw_mol
    smol_mol = GeometricMol.from_rdkit(mols[0])
    return smol_mol


def raw_to_smol_batch(raw_data):
    smol_mols = [raw_to_smol_mol(raw_mol) for raw_mol in raw_data]
    batch = GeometricMolBatch.from_list(smol_mols)
    return batch


def process_dataset(raw_filepath, save_filepath):
    raw_dataset = read_from_file(raw_filepath)
    smol_batch = raw_to_smol_batch(raw_dataset)
    dataset_bytes = smol_batch.to_bytes()
    save_filepath.write_bytes(dataset_bytes)


def main(args):
    data_path = Path(args.data_path)

    raw_data_path = data_path / args.raw_data_folder
    raw_train_path = raw_data_path / RAW_TRAIN_FILE
    raw_val_path = raw_data_path / RAW_VAL_FILE
    raw_test_path = raw_data_path / RAW_TEST_FILE

    assert raw_train_path.exists()
    assert raw_val_path.exists()
    assert raw_test_path.exists()

    save_data_path = data_path / args.save_data_folder
    save_data_path.mkdir(parents=True, exist_ok=True)
    save_train_path = save_data_path / SAVE_TRAIN_FILE
    save_val_path = save_data_path / SAVE_VAL_FILE
    save_test_path = save_data_path / SAVE_TEST_FILE

    print("Processing train dataset...")
    process_dataset(raw_train_path, save_train_path)
    print("Train dataset complete.")

    print("Processing val dataset...")
    process_dataset(raw_val_path, save_val_path)
    print("Val dataset complete.")

    print("Processing test dataset...")
    process_dataset(raw_test_path, save_test_path)
    print("Test dataset complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--raw_data_folder", type=str, default=DEFAULT_RAW_DATA_FOLDER)
    parser.add_argument("--save_data_folder", type=str, default=DEFUALT_SAVE_DATA_FOLDER)

    args = parser.parse_args()
    main(args)
