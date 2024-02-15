from lightning.pytorch.cli import LightningCLI

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
        parser.add_argument("--wandblog", type=str, default="")
        parser.add_argument("--log_freq", type=int, default=500)
        parser.add_argument("--log_graph", type=bool, default=False)
        parser.add_argument("--project", type=str)

    def before_instantiate_classes(self):
        for k, v in self.config.items():
            if "set_float32_matmul_precision" in k:
                if v:
                    torch.set_float32_matmul_precision("medium")
