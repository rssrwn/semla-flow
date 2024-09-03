import argparse
from functools import partial
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import semlaflow.scriptutil as util
from semlaflow.data.datamodules import GeometricInterpolantDM
from semlaflow.data.datasets import GeometricDataset
from semlaflow.data.interpolate import GeometricInterpolant, GeometricNoiseSampler
from semlaflow.models.fm import Integrator, MolecularCFM
from semlaflow.models.semla import EquiInvDynamics, SemlaGenerator

DEFAULT_DATASET = "geom-drugs"
DEFAULT_ARCH = "semla"

DEFAULT_D_MODEL = 256  # 384
DEFAULT_N_LAYERS = 4  # 12
DEFAULT_D_MESSAGE = 64  # 128
DEFAULT_D_EDGE = 64  # 128
DEFAULT_N_COORD_SETS = 32  # 64
DEFAULT_N_ATTN_HEADS = 16  # 32
DEFAULT_D_MESSAGE_HIDDEN = 64  # 128
DEFAULT_COORD_NORM = "length"
DEFAULT_SIZE_EMB = 32  # 64

DEFAULT_MAX_ATOMS = 128  # 256

DEFAULT_EPOCHS = 200
DEFAULT_LR = 0.0003
DEFAULT_BATCH_COST = 4096
DEFAULT_ACC_BATCHES = 1
DEFAULT_GRADIENT_CLIP_VAL = 1.0
DEFAULT_TYPE_LOSS_WEIGHT = 0.2
DEFAULT_BOND_LOSS_WEIGHT = 1.0
DEFAULT_CHARGE_LOSS_WEIGHT = 1.0
DEFAULT_CATEGORICAL_STRATEGY = "uniform-sample"
DEFAULT_LR_SCHEDULE = "constant"
DEFAULT_WARM_UP_STEPS = 10000
DEFAULT_BUCKET_COST_SCALE = "linear"

DEFAULT_N_VALIDATION_MOLS = 1000  # 2000
DEFAULT_VALIDATION_EPOCHS = 10
DEFAULT_NUM_INFERENCE_STEPS = 100
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1
DEFAULT_COORD_NOISE_STD_DEV = 0.2
DEFAULT_TYPE_DIST_TEMP = 1.0
DEFAULT_TIME_ALPHA = 2.0
DEFAULT_TIME_BETA = 1.0
DEFAULT_OPTIMAL_TRANSPORT = "equivariant"


# bfloat16 training produced significantly worse models than full so use default 16-bit instead
def get_precision(args):
    return "32"


def build_model(args, dm, vocab):
    # Get hyperparameeters from the datamodule, pass these into the model to be saved
    hparams = {
        "epochs": args.epochs,
        "gradient_clip_val": args.gradient_clip_val,
        "dataset": args.dataset,
        "precision": get_precision(args),
        "architecture": args.arch,
        **dm.hparams
    }

    # Add 1 for the time (0 <= t <= 1 for flow matching)
    n_atom_feats = vocab.size + 1
    n_bond_types = util.get_n_bond_types(args.categorical_strategy)

    if args.arch == "semla":
        dynamics = EquiInvDynamics(
            args.d_model,
            args.d_message,
            args.n_coord_sets,
            args.n_layers,
            n_attn_heads=args.n_attn_heads,
            d_message_hidden=args.d_message_hidden,
            d_edge=args.d_edge,
            bond_refine=True,
            self_cond=args.self_condition,
            coord_norm=args.coord_norm
        )
        egnn_gen = SemlaGenerator(
            args.d_model,
            dynamics,
            vocab.size,
            n_atom_feats,
            d_edge=args.d_edge,
            n_edge_types=n_bond_types,
            self_cond=args.self_condition,
            size_emb=args.size_emb,
            max_atoms=args.max_atoms
        )

    elif args.arch == "eqgat":
        from semlaflow.models.eqgat import EqgatGenerator

        # Hardcode for now since we only need one model size
        d_model_eqgat = 256
        n_equi_feats_eqgat = 256
        n_layers_eqgat = 12
        d_edge_eqgat = 128

        egnn_gen = EqgatGenerator(
            d_model_eqgat,
            n_layers_eqgat,
            n_equi_feats_eqgat,
            vocab.size,
            n_atom_feats,
            d_edge_eqgat,
            n_bond_types
        )

    elif args.arch == "egnn":
        from semlaflow.models.egnn import VanillaEgnnGenerator

        egnn_gen = VanillaEgnnGenerator(
            args.d_model,
            args.n_layers,
            vocab.size,
            n_atom_feats,
            d_edge=args.d_edge,
            n_edge_types=n_bond_types
        )

    else:
        raise ValueError(f"Unknown architecture '{args.arch}'. ")

    if args.dataset == "qm9":
        coord_scale = util.QM9_COORDS_STD_DEV
    elif args.dataset == "geom-drugs":
        coord_scale = util.GEOM_COORDS_STD_DEV
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    type_mask_index = None
    bond_mask_index = None

    if args.categorical_strategy == "mask":
        type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0]
        bond_mask_index = util.BOND_MASK_INDEX
        train_strategy = "mask"
        sampling_strategy = "mask"

    elif args.categorical_strategy == "uniform-sample":
        train_strategy = "ce"
        sampling_strategy = "uniform-sample"

    elif args.categorical_strategy == "dirichlet":
        train_strategy = "ce"
        sampling_strategy = "dirichlet"

    else:
        raise ValueError(f"Interpolation '{args.categorical_strategy}' is not supported. "
                         + "Supported are: `mask`, `uniform-sample` and `dirichlet`")

    train_steps = util.calc_train_steps(dm, args.epochs, args.acc_batches)
    train_smiles = None if args.trial_run else [mols.str_id for mols in dm.train_dataset]

    print(f"Total training steps {train_steps}")

    integrator = Integrator(
        args.num_inference_steps,
        type_strategy=sampling_strategy,
        bond_strategy=sampling_strategy,
        cat_noise_level=args.cat_sampling_noise_level,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index
    )

    fm_model = MolecularCFM(
        egnn_gen,
        vocab,
        args.lr,
        integrator,
        coord_scale=coord_scale,
        type_strategy=train_strategy,
        bond_strategy=train_strategy,
        type_loss_weight=args.type_loss_weight,
        bond_loss_weight=args.bond_loss_weight,
        charge_loss_weight=args.charge_loss_weight,
        pairwise_metrics=False,
        use_ema=args.use_ema,
        compile_model=False,
        self_condition=args.self_condition,
        distill=False,
        lr_schedule=args.lr_schedule,
        warm_up_steps=args.warm_up_steps,
        total_steps=train_steps,
        train_smiles=train_smiles,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        **hparams
    )
    return fm_model


def build_dm(args, vocab):
    if args.dataset == "qm9":
        coord_std = util.QM9_COORDS_STD_DEV
        padded_sizes = util.QM9_BUCKET_LIMITS

    elif args.dataset == "geom-drugs":
        coord_std = util.GEOM_COORDS_STD_DEV
        padded_sizes = util.GEOM_DRUGS_BUCKET_LIMITS

    else:
        raise ValueError(f"Unknown dataset {args.dataset}. Available datasets are `qm9` and `geom-drugs`.")

    data_path = Path(args.data_path)

    n_bond_types = util.get_n_bond_types(args.categorical_strategy)
    transform = partial(util.mol_transform, vocab=vocab, n_bonds=n_bond_types, coord_std=coord_std)

    # Load generated dataset with different transform fn if we are distilling a model
    # if args.distill:
    #     distill_transform = partial(util.distill_transform, coord_std=coord_std)
    #     train_dataset = GeometricDataset.load(data_path / "distill.smol", transform=distill_transform)
    # else:
    #     train_dataset = GeometricDataset.load(data_path / "train.smol", transform=transform)

    train_dataset = GeometricDataset.load(data_path / "train.smol", transform=transform)
    val_dataset = GeometricDataset.load(data_path / "val.smol", transform=transform)
    val_dataset = val_dataset.sample(args.n_validation_mols)

    type_mask_index = None
    bond_mask_index = None

    if args.categorical_strategy == "mask":
        type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0]
        bond_mask_index = util.BOND_MASK_INDEX
        categorical_interpolation = "unmask"
        categorical_noise = "mask"

    elif args.categorical_strategy == "uniform-sample":
        categorical_interpolation = "unmask"
        categorical_noise = "uniform-sample"

    elif args.categorical_strategy == "dirichlet":
        categorical_interpolation = "dirichlet"
        categorical_noise = "uniform-dist"

    else:
        raise ValueError(f"Interpolation '{args.categorical_strategy}' is not supported. "
                         + "Supported are: `mask`, `uniform-sample` and `dirichlet`")

    scale_ot = False
    batch_ot = False
    equivariant_ot = False

    if args.optimal_transport == "batch":
        batch_ot = True
    elif args.optimal_transport == "equivariant":
        equivariant_ot = True
    elif args.optimal_transport == "scale":
        scale_ot = True
        equivariant_ot = True
    elif args.optimal_transport not in ["None", "none", None]:
        raise ValueError(f"Unknown value for optimal_transport '{args.optimal_transport}'. "
                         + "Acceted values: `batch`, `equivariant` and `scale`.")

    # train_fixed_time = 0.5 if args.distill else None
    train_fixed_time = None

    prior_sampler = GeometricNoiseSampler(
        vocab.size,
        n_bond_types,
        coord_noise="gaussian",
        type_noise=categorical_noise,
        bond_noise=categorical_noise,
        scale_ot=scale_ot,
        zero_com=True,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index
    )
    train_interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation="linear",
        type_interpolation=categorical_interpolation,
        bond_interpolation=categorical_interpolation,
        coord_noise_std=args.coord_noise_std_dev,
        type_dist_temp=args.type_dist_temp,
        equivariant_ot=equivariant_ot,
        batch_ot=batch_ot,
        time_alpha=args.time_alpha,
        time_beta=args.time_beta,
        fixed_time=train_fixed_time
    )
    eval_interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation="linear",
        type_interpolation=categorical_interpolation,
        bond_interpolation=categorical_interpolation,
        equivariant_ot=False,
        batch_ot=False,
        fixed_time=0.9
    )

    dm = GeometricInterpolantDM(
        train_dataset,
        val_dataset,
        None,
        args.batch_cost,
        train_interpolant=train_interpolant,
        val_interpolant=eval_interpolant,
        test_interpolant=None,
        bucket_limits=padded_sizes,
        bucket_cost_scale=args.bucket_cost_scale,
        pad_to_bucket=False
    )
    return dm


def build_trainer(args):
    epochs = 1 if args.trial_run else args.epochs
    log_steps = 1 if args.trial_run else 50
    val_check_epochs = 1 if args.trial_run else args.val_epochs

    project_name = f"{util.PROJECT_PREFIX}-{args.dataset}"
    precision = get_precision(args)
    print(f"Using precision '{precision}'")

    logger = WandbLogger(project=project_name, save_dir="wandb", log_model=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpointing = ModelCheckpoint(
        every_n_epochs=val_check_epochs,
        monitor="val-validity",
        mode="max",
        save_last=True
    )

    # No logger if doing a trial run
    logger = None if args.trial_run else logger

    trainer = L.Trainer(
        min_epochs=epochs,
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=log_steps,
        accumulate_grad_batches=args.acc_batches,
        gradient_clip_val=args.gradient_clip_val,
        check_val_every_n_epoch=val_check_epochs,
        callbacks=[lr_monitor, checkpointing],
        precision=precision
    )
    return trainer


def main(args):
    # Set some useful torch properties
    # Float32 precision should only affect computation on A100 and should in theory be a lot faster than the default
    # Increasing the cache size is required since the model will be compiled seperately for each bucket
    torch.set_float32_matmul_precision("high")
    # torch._dynamo.config.cache_size_limit = util.COMPILER_CACHE_SIZE
    # print(f"Set torch compiler cache size to {torch._dynamo.config.cache_size_limit}")

    L.seed_everything(12345)
    util.disable_lib_stdout()
    util.configure_fs()

    print("Building model vocab...")
    vocab = util.build_vocab()
    print("Vocab complete.")

    print("Loading datamodule...")
    dm = build_dm(args, vocab)
    print("Datamodule complete.")

    print("Building equinv model...")
    model = build_model(args, dm, vocab)
    print("Model complete.")

    trainer = build_trainer(args)

    print("Fitting datamodule to model...")
    trainer.fit(model, datamodule=dm)
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Setup args
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--trial_run", action="store_true")

    # Model args
    parser.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--d_message", type=int, default=DEFAULT_D_MESSAGE)
    parser.add_argument("--d_edge", type=int, default=DEFAULT_D_EDGE)
    parser.add_argument("--n_coord_sets", type=int, default=DEFAULT_N_COORD_SETS)
    parser.add_argument("--n_attn_heads", type=int, default=DEFAULT_N_ATTN_HEADS)
    parser.add_argument("--d_message_hidden", type=int, default=DEFAULT_D_MESSAGE_HIDDEN)
    parser.add_argument("--coord_norm", type=str, default=DEFAULT_COORD_NORM)
    parser.add_argument("--size_emb", type=int, default=DEFAULT_SIZE_EMB)
    parser.add_argument("--max_atoms", type=int, default=DEFAULT_MAX_ATOMS)
    parser.add_argument("--arch", type=str, default=DEFAULT_ARCH)

    # Training args
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--acc_batches", type=int, default=DEFAULT_ACC_BATCHES)
    parser.add_argument("--gradient_clip_val", type=float, default=DEFAULT_GRADIENT_CLIP_VAL)
    parser.add_argument("--type_loss_weight", type=float, default=DEFAULT_TYPE_LOSS_WEIGHT)
    parser.add_argument("--bond_loss_weight", type=float, default=DEFAULT_BOND_LOSS_WEIGHT)
    parser.add_argument("--charge_loss_weight", type=float, default=DEFAULT_CHARGE_LOSS_WEIGHT)
    parser.add_argument("--categorical_strategy", type=str, default=DEFAULT_CATEGORICAL_STRATEGY)
    parser.add_argument("--lr_schedule", type=str, default=DEFAULT_LR_SCHEDULE)
    parser.add_argument("--warm_up_steps", type=int, default=DEFAULT_WARM_UP_STEPS)
    parser.add_argument("--bucket_cost_scale", type=str, default=DEFAULT_BUCKET_COST_SCALE)
    parser.add_argument("--no_ema", action="store_false", dest="use_ema")
    parser.add_argument("--self_condition", action="store_true")
    # parser.add_argument("--mixed_precision", action="store_true")
    # parser.add_argument("--compile_model", action="store_true")
    # parser.add_argument("--distill", action="store_true")

    # Flow matching and sampling args
    parser.add_argument("--val_epochs", type=int, default=DEFAULT_VALIDATION_EPOCHS)
    parser.add_argument("--n_validation_mols", type=int, default=DEFAULT_N_VALIDATION_MOLS)
    parser.add_argument("--num_inference_steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS)
    parser.add_argument("--cat_sampling_noise_level", type=int, default=DEFAULT_CAT_SAMPLING_NOISE_LEVEL)
    parser.add_argument("--coord_noise_std_dev", type=float, default=DEFAULT_COORD_NOISE_STD_DEV)
    parser.add_argument("--type_dist_temp", type=float, default=DEFAULT_TYPE_DIST_TEMP)
    parser.add_argument("--time_alpha", type=float, default=DEFAULT_TIME_ALPHA)
    parser.add_argument("--time_beta", type=float, default=DEFAULT_TIME_BETA)
    parser.add_argument("--optimal_transport", type=str, default=DEFAULT_OPTIMAL_TRANSPORT)

    parser.set_defaults(
        trial_run=False,
        use_ema=True,
        self_condition=True,
        # compile_model=False,
        # mixed_precision=False,
        # distill=False
    )

    args = parser.parse_args()
    main(args)
