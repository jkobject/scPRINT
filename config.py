from dataclasses import dataclass


@dataclass
class Config:
    # Data
    data_dir: str = "data"
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    sample_submission_file: str = "sample_submission.csv"
    # Model
    model_dir: str = "models"
    model_name: str = "model.pkl"
    # Log
    log_dir: str = "logs"
    log_file: str = "log.txt"
    # Seed
    seed: int = 42
    # Training
    n_folds: int = 5
    num_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    # Inference
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Logging
    verbose: bool = True
    do_wandb: bool = False
    # Debug
    debug: bool = False

class Paths:
    embeddings: str = "./data/temp/embeddings.parquet"
    collection_to_use: str = "preprocessed dataset"
