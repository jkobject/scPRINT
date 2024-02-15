from dataclasses import dataclass


@dataclass
class Config:
    # Data
    data_dir: str = "data"
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    gene_embeddings: str = "./data/temp/embeddings.parquet"
    collection_name: str = "preprocessed dataset"
    gene_position_tolerance: int = 10000
    organisms = ["NCBITaxon:9606"]
    # Model
    d_model: int = 128
    model_name: str = "model.pkl"
    do_gene_pos: bool = True
    # Log
    log_dir: str = "./data/tensorboard"
    project: str = "scprint_test"
    log_file: str = "log.txt"
    # Seed
    seed: int = 42
    # Training
    set_float32_precision: bool = True
    n_folds: int = 5
    num_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    # Inference
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Logging
    verbose: bool = True
    logger: str = "wandb"
    # Debug
    debug: bool = False


class Paths:
    embeddings: str = "./data/temp/embeddings.parquet"
    collection_to_use: str = "preprocessed dataset"
