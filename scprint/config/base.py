from dataclasses import dataclass


@dataclass
class MNISTConf:
    batch_size: int = 64
    test_batch_size: int = 1000
    epochs: int = 14
    no_cuda: bool = False
    dry_run: bool = False
    seed: int = 1
    log_interval: int = 10
    save_model: bool = False
    checkpoint_name: str = "unnamed.pt"
    adadelta: AdadeltaConf = AdadeltaConf()
    steplr: StepLRConf = StepLRConf(
        step_size=1
    )  # we pass a default for step_size since it is required, but missing a default in PyTorch (and consequently in hydra-torch)


@dataclass
class BaseConfig:
    seed: int = 42


class TrainerConfig:
    mask_ratio: float = 0.4  # Default mask ratio
    epochs: int = 15  # Default number of epochs for fine-tuning
    lr: float = 1e-4  # Default learning rate for fine-tuning

    batch_size: int = 64  # Default batch size for fine-tuning
    dropout: float = 0.2  # Default dropout rate during model fine-tuning
    schedule_ratio: float = 0.9  # Default rate for learning rate decay


@dataclass
class scGPTModelConfig:
    GEPC: bool = True  # Gene expression modelling for cell objective
    ecs_thres: float = (
        0.8  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    )
    dab_weight: float = 1.0  # DAR objective weight for batch correction

    n_bins: int = 51  # Default number of bins for value binning in data pre-processing
    layer_size: int = 128
    nlayers: int = 4
    nhead: int = (
        4  # if load model, batch_size, layer_size, nlayers, nhead will be ignored
    )

    fast_transformer: bool = True  # Default setting
    pre_norm: bool = False  # Default setting
    amp: bool = True  # Default setting: Automatic Mixed Precision
    special_tokens: list[str] = ["<pad>", "<unk>", "<mask>"]
    per_seq_batch_sample = True
    amp
    


class ExperimentConfig:
    seed: int = 42
    do_train: bool = (
        True  # Flag to indicate whether to do update model parameters during training
    )
    load_model: str = "../save/scGPT_human"  # Path to pre-trained model
    dataset_name: str = "PBMC_10K"  # Dataset name
    log_interval: int = 100  # Default log interval
    save_eval_interval: int = 5  # Default model evaluation interval
    mask_value = -1
    pad_value = -2


class PreprocessorConfig:
    use_key = "X"  # the key in adata.layers to use as raw data
    filter_gene_by_counts = 3  # step 1
    filter_cell_by_counts = False  # step 2
    normalize_total = 1e4  # 3. whether to normalize the raw data and to what sum
    result_normed_key = (
        "X_normed"  # the key in adata.layers to store the normalized data
    )
    log1p = True  # 4. whether to log1p the normalized data
    result_log1p_key = "X_log1p"
    subset_hvg = 2000  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor = "seurat_v3"
