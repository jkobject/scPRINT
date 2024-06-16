from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from .trainer import TrainingMode
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import StochasticWeightAveraging
import torch


class MyCLI(LightningCLI):
    """
    MyCLI is a subclass of LightningCLI to add some missing params
    and create bindings between params of the model and the data.
    """        
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(EarlyStopping, "scprint_early_stopping")
        parser.set_defaults({"scprint_early_stopping.monitor": "val_loss", "scprint_early_stopping.patience": 3})
        parser.add_lightning_class_args(LearningRateMonitor, "scprint_learning_rate_monitor")
        parser.set_defaults({"scprint_learning_rate_monitor.logging_interval": "epoch"})
        parser.add_lightning_class_args(TrainingMode, "scprint_training")
        parser.link_arguments(
            "data.gene_pos", "model.gene_pos_enc", apply_on="instantiate"
        )
        parser.link_arguments("data.genes", "model.genes", apply_on="instantiate")
        parser.link_arguments(
            "data.decoders", "model.label_decoders", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.labels_hierarchy", "model.labels_hierarchy", apply_on="instantiate"
        )
        parser.link_arguments("data.classes", "model.classes", apply_on="instantiate")
        parser.link_arguments(
            "data.gene_embeddings", "model.precpt_gene_emb", apply_on="parse"
        )
        parser.link_arguments(
            "data.num_datasets", "model.num_batch_labels", apply_on="instantiate"
        )
        parser.add_argument("--set_float32_matmul_precision", type=bool, default=False)
        parser.add_argument("--wandblog", type=str, default="")
        parser.add_argument("--log_freq", type=int, default=500)
        parser.add_argument("--log_graph", type=bool, default=False)
        parser.add_argument("--project", type=str)

    def before_instantiate_classes(self):
        for k, v in self.config.items():
            if "set_float32_matmul_precision" in k:
                if v:
                    torch.set_float32_matmul_precision("medium")
