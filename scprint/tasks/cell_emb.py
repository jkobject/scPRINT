import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import torch

from scdataloader.data import SimpleAnnDataset
from scdataloader import Collator
from scprint.model import utils

from torch.utils.data import DataLoader

import pandas as pd

from lightning.pytorch import Trainer


class Embedder:
    def __init__(
        self,
        model,
        batch_size=64,
        num_workers=8,
        how="most expr",
        max_len=1000,
        add_zero_genes=100,
        precision="16-mixed",
        organisms=[
            "NCBITaxon:9606",
        ],
        pred_embedding=[
            "cell_type_ontology_term_id",
            "disease_ontology_term_id",
            "self_reported_ethnicity_ontology_term_id",
            "sex_ontology_term_id",
        ],
        model_name="scprint",
        output_expression="sample",  # one of "all" "sample" "none"
        plot_corr_size=64,
    ):
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.how = how
        self.max_len = max_len
        self.add_zero_genes = add_zero_genes
        self.organisms = organisms
        self.model.pred_embedding = pred_embedding
        self.model_name = model_name
        self.output_expression = output_expression
        self.plot_corr_size = plot_corr_size
        self.trainer = Trainer(precision=precision)
        # subset_hvg=1000, use_layer='counts', is_symbol=True,force_preprocess=True, skip_validate=True)

    def __call__(self, adata):
        # Add at least the organism you are working with
        adataset = SimpleAnnDataset(adata, obs_to_output=["organism_ontology_term_id"])
        col = Collator(
            organisms=self.organisms,
            valid_genes=self.model.genes,
            how=self.how,
            max_len=self.max_len,
            add_zero_genes=self.add_zero_genes,
        )
        dataloader = DataLoader(
            adataset,
            collate_fn=col,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        self.trainer.predict(self.model, dataloader)
        try:
            mdir = (
                self.model.logger.save_dir
                if self.model.logger.save_dir is not None
                else "/tmp"
            )
        except:
            mdir = "/tmp"

        pred_adata = sc.read_h5ad(
            mdir + "/step_" + str(self.model.global_step) + "_" + ".h5ad"
        )
        if self.output_expression == "all":
            adata.obsm["scprint_mu"] = self.model.expr_pred[0]
            adata.obsm["scprint_theta"] = self.model.expr_pred[1]
            adata.obsm["scprint_pi"] = self.model.expr_pred[2]
        elif self.output_expression == "sample":
            adata.obsm["scprint_expr"] = utils.zinb_sample(
                self.model.expr_pred[0],
                self.model.expr_pred[1],
                self.model.expr_pred[2],
            )
        elif self.output_expression == "old":
            expr = np.array(self.model.expr_pred[0])
            expr[
                np.random.binomial(
                    1,
                    p=np.array(
                        torch.nn.functional.sigmoid(
                            self.model.expr_pred[2].to(torch.float32)
                        )
                    ),
                ).astype(bool)
            ] = 0
            expr[expr <= 0.3] = 0
            expr[(expr >= 0.3) & (expr <= 1)] = 1
            adata.obsm["scprint_expr"] = expr.astype(int)
        adata.obsm["scprint_pos"] = self.model.pos

        pred_adata.obs.index = adata.obs.index
        adata.obsm["scprint_umap"] = pred_adata.obsm["X_umap"]
        adata.obsm[self.model_name] = pred_adata.X
        pred_adata.obs.index = adata.obs.index
        adata.obs = pd.concat([adata.obs, pred_adata.obs], axis=1)

        # Compute correlation coefficient
        if self.plot_corr_size > 0:
            random_indices = np.random.randint(
                low=0, high=adata.shape[0], size=self.plot_corr_size
            )
            pos = adata.obsm["scprint_pos"][random_indices]
            X = adata.X[:, adata.var.index.isin(self.model.genes)][random_indices]
            corr_coef = np.corrcoef(
                adata.obsm["scprint_expr"][random_indices].numpy(),
                X[
                    np.array(list(range(self.plot_corr_size)))[:, None], pos.numpy()
                ].toarray(),
            )[:, :]

            # Plot correlation coefficient
            plt.figure(figsize=(10, 5))
            plt.imshow(corr_coef, cmap="coolwarm", interpolation="none")
            plt.colorbar()
            plt.title('Correlation Coefficient of expr and i["x"]')
            plt.show()
        return adata
