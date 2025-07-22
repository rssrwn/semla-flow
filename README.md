# SemlaFlow - Efficient 3D Molecular Generation with Latent Attention and Equivariant Flow Matching

This project creates a novel equivariant attention-based message passing architecture, Semla, for molecular design and dynamics tasks. We train a molecular generation model, SemlaFlow, using flow matching with optimal transport to generate realistic 3D molecular structures.


## Evaluation Metrics

**A note on evaluation metrics for 3D molecular generative models** - As we pointed out in the paper, many of the existing commonly used metrics for this task are not helpful for assessing the quality of the generated molecules. In addition to the metrics we have proposed here, some recent papers [1][2] have done a great job thoroughly evaluating the quality of generated structure, which may also be of interest.


## Installation

All of the code was run using a mamba/conda environment. You can of course use a different environment manager; all core requirements are contained in the `environment.yaml` file. Using mamba/conda you can recreate the environment as follows:
1. `mamba env create --file environment.yaml`
2. `mamba activate semlaflow`

For developing (and to run the notebooks) you will also need to install the extra requirements:
3. `pip install -r extra_requirements.txt`


## Datasets

For ease-of-use we have provided the processed data files in a Google drive [here](https://drive.google.com/drive/folders/1rHi5JzN05bsGRGQUcWRmDu-Ilfoa9EAT?usp=sharing). Copy the folder called `smol` from the QM9 or GEOM drugs folders and point to the `smol` folder when running the scripts. For example, pass `--data_path path/to/data/qm9/smol` to the script you wish to run.


### Data Prep

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
* `train` - Trains a SemlaFlow model on preprocessed data
* `evaluate` - Evaluates a trained model and prints the results
* `predict` - Runs the sampling for a trained model and saves the generated molecules

Each script can be run as follows (where `<script>` is replaced by the script name above without `.py`): `python -m semlaflow.<script> --data_path <path/to/data> <other_args>`

See the bottom of each script for a full list of the arguments available. Default paramaters for those arguments are also given as global declarations at the top of each file. The default arguments in the training script are for GEOM Drugs. To train on QM9 we use a `bond_loss_weight` of 0.5, 2000 `warm_up_steps` and usually 300 `epochs`.

### Models

We also provide pretrained model checkpoints for our headline QM9 and GEOM drugs models [here](https://drive.google.com/drive/folders/1rHi5JzN05bsGRGQUcWRmDu-Ilfoa9EAT?usp=sharing). If you wish to evaluate one of these models pass the checkpoint to the evaluate script. For example `--ckpt_path path/to/models/qm9.ckpt`

### Tests

The tests are quite sparse and only test the core functionality of the util functions used throughout the model and the molecular representations. 

To run all tests `python -m unittest -v tests/*.py`

Specific test modules can also be run individually. Eg. `python -m unittest -v tests.functional`


## Contact

If you find a problem with the code feel free to make a PR. If you have questions or other issues with the code you can email me directly -> rossir [at] chalmers [dot] se


## Citation

```
@inproceedings{irwin2025semlaflow,
  title={SemlaFlow -- Efficient 3D Molecular Generation with Latent Attention and Equivariant Flow Matching},
  author={Ross Irwin and Alessandro Tibo and Jon Paul Janet and Simon Olsson},
  booktitle={The 28th International Conference on Artificial Intelligence and Statistics},
  year={2025},
  url={https://openreview.net/forum?id=bee2G6pEh0}
}
```

## Additional References

Additionally, as mentioned above, some other recent papers have done an excellent job benchmarking 3D molecular generative models, which may also be of interest:
* [1] Nikitin, Filipp, et al. "GEOM-Drugs Revisited: Toward More Chemically Accurate Benchmarks for 3D Molecule Generation." arXiv preprint arXiv:2505.00169 (2025).
* [2] Buttenschoen, Martin, et al. "An evaluation of unconditional 3D molecular generation methods." arXiv preprint arXiv:2505.00518 (2025).
