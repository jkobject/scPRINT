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

from scipy.stats import spearmanr


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
        self.precision = precision
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
            sc.pp.highly_variable_genes(
                adata, n_top_genes=self.max_len * 2, flavor="seurat_v3"
            )
            highly_variable = adata.var.index[adata.var.highly_variable].tolist()
            random_indices = np.random.randint(
                low=0, high=adata.shape[0], size=self.plot_corr_size
            )
            adataset = SimpleAnnDataset(
                adata[random_indices], obs_to_output=["organism_ontology_term_id"]
            )
            col = Collator(
                organisms=self.organisms,
                valid_genes=self.model.genes,
                how="some",
                genelist=highly_variable,
            )
            dataloader = DataLoader(
                adataset,
                collate_fn=col,
                batch_size=self.plot_corr_size,
                num_workers=self.num_workers,
                shuffle=False,
            )
            # self.trainer.num_predict_batches = 1
            self.trainer.predict(self.model, dataloader)

            res = self.model.expr_pred
            # pos = adata.obsm["scprint_pos"][random_indices]
            out = utils.zinb_sample(
                res[0],
                res[1],
                res[2],
            ).numpy()
            for i in dataloader:
                break
            try:
                mean_expr = pd.read_parquet("../../data/avg_expr.parquet")
                genes_used = [self.model.genes[int(i)] for i in self.model.pos[0]]
                mean_expr = mean_expr[mean_expr.index.isin(genes_used)][
                    ["avg_expr", "avg_expr_wexpr"]
                ].values
                out = np.hstack([out.T, mean_expr])
            except:
                print(
                    "cannot read the mean expr file under scprint/data/avg_expr.parquet"
                )
                out = out.T
                mean_expr = None

            corr_coef, p_value = spearmanr(
                out,
                i["x"].T,
            )
            corr_coef[p_value > 0.05] = 0
            # corr_coef[]
            # only on non zero values,
            # compare a1-b1 corr with a1-b(n) corr. should be higher

            # Plot correlation coefficient
            plt.figure(figsize=(10, 5))
            plt.imshow(
                corr_coef, cmap="coolwarm", interpolation="none", vmin=-1, vmax=1
            )
            plt.colorbar()
            plt.title('Correlation Coefficient of expr and i["x"]')
            plt.show()
            expr = np.array(res[0])
            expr[
                np.random.binomial(
                    1,
                    p=np.array(torch.nn.functional.sigmoid(res[2].to(torch.float32))),
                ).astype(bool)
            ] = 0
            corr_coef, p_value = spearmanr(
                np.hstack([expr.T, mean_expr]) if mean_expr is not None else expr.T,
                i["x"].T,
            )
            corr_coef[p_value > 0.05] = 0
            # corr_coef[]
            # only on non zero values,
            # compare a1-b1 corr with a1-b(n) corr. should be higher

            # Plot correlation coefficient
            plt.figure(figsize=(10, 5))
            plt.imshow(
                corr_coef, cmap="coolwarm", interpolation="none", vmin=-1, vmax=1
            )
            plt.colorbar()
            plt.title('Correlation Coefficient of expr and i["x"]')
            plt.show()
        return adata
