from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger

import torch


class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.gene_pos", "model.gene_pos_enc", apply_on="instantiate"
        )
        parser.link_arguments("data.genes", "model.genes", apply_on="instantiate")
        parser.link_arguments(
            "data.decoders", "model.label_decoders", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.cls_hierarchy", "model.cls_hierarchy", apply_on="instantiate"
        )
        parser.link_arguments("data.labels", "model.labels", apply_on="instantiate")
        parser.link_arguments(
            "data.gene_embeddings", "model.precpt_gene_emb", apply_on="parse"
        )
        parser.add_argument("--set_float32_matmul_precision", type=bool, default=False)
        parser.add_argument("--project", type=str)

    def before_instantiate_classes(self):
        for k, v in self.config.items():
            if "set_float32_matmul_precision" in k:
                if v:
                    torch.set_float32_matmul_precision("medium")


class MySaveConfig(SaveConfigCallback):
    def __init__(self, *args, wandblog="all", log_freq=500, log_graph=False, **kwargs):
        self.wandblog = wandblog
        self.wandblog_freq = log_freq
        self.wandblog_graph = log_graph
        super().__init__(*args, **kwargs)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if type(trainer.logger) is WandbLogger:
            trainer.logger.watch(
                pl_module,
                log=self.wandblog,
                log_freq=self.wandblog_freq,
                log_graph=self.wandblog_graph,
            )
        return super().setup(trainer, pl_module, stage)
