# SemlaFlow - Efficient Molecular Generation with Flow Matching and Semla

This project creates a novel equivariant attention-based message passing architecture, Semla, for molecular design and dynamics tasks. We train a molecular generation model, SemlaFlow, using flow matching with optimal transport to generate realistic 3D molecular structures.


## Installation

All of the code was run using a mamba/conda environment. You can of course use a different environment manager; all core requirements (other than cxx-compiler) are contained in the `requirements.txt` file. Using mamba/conda you can recreate the environment as follows:
1. `mamba create --name equinv python=3.11`
2. `mamba activate equinv`
3. `mamba install -c conda-forge cxx-compiler`
4. `pip install -r requirements.txt`

For developing (and to run the notebooks) you will also need to install the extra requirements:
5. `pip install -r extra_requirements.txt`


## Datasets

We copied the code from MiDi (https://github.com/cvignac/MiDi) to download the QM9 dataset and create the data splits. We provide the code to do this, as well as create the _Smol_ internal dataset representation used for training in the `notebooks/qm9.ipynb` notebook.

For GEOM Drugs we also follow the URLs provided in the MiDi repo. GEOM Drugs is preprocessed using the `preprocess.py` script. GEOM Drugs URLs from MiDi are as follows:
* train: https://drive.switch.ch/index.php/s/UauSNgSMUPQdZ9v
* validation: https://drive.switch.ch/index.php/s/YNW5UriYEeVCDnL
* test: https://drive.switch.ch/index.php/s/GQW9ok7mPInPcIo


## Running

Once you have created and activated the environment successfully, you can run the code.

### Scripts

We provide 4 scripts in the repository:
* `preprocess` - Used for preprocessing larger datasets into the internal representation used by the model for training
* `train` - Trains a MolFlow model on preprocessed data
* `evaluate` - Evaluates a trained model and prints the results
* `predict` - Runs the sampling for a trained model and saves the generated molecules

Each script can be run as follows (where `<script>` is replaced by the script name above without `.py`): `python -m semlaflow.<script> --data_path <path/to/data> <other_args>`

See the bottom of each script for a full list of the arguments available. Default paramaters for those arguments are also given as global declarations at the top of each file.

### Tests

The tests are quite sparse and only test the core functionality of the util functions used throughout the model and the molecular representations. 

To run all tests `python -m unittest -v tests/*.py`

Specific test modules can also be run individually. Eg. `python -m unittest -v tests.functional`


## Contact

If you find a problem with the code feel free to make a PR. If you have questions or other issues with the code you can email me directly -> rossir [at] chalmers [dot] se
