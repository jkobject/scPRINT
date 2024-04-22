import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import torch

from scdataloader.data import SimpleAnnDataset
from scdataloader import Collator
from scprint.model import utils
import bionty as bt
from torch.utils.data import DataLoader

import pandas as pd

from lightning.pytorch import Trainer


from typing import List
from anndata import AnnData
from scprint.tasks import compute_corr

from scipy.stats import spearmanr
import os


class Denoiser:
    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int = 10,
        num_workers: int = 1,
        max_len: int = 20000,
        precision: str = "16-mixed",
        organisms: List[str] = [
            "NCBITaxon:9606",
        ],
        plot_corr_size: int = 64,
        doplot: bool = True,
        predict_depth_mult: int = 4,
    ):
        """
        Embedder a class to embed and annotate cells using a model

        Args:
            model (torch.nn.Module): The model to be used for embedding and annotating cells.
            batch_size (int, optional): The size of the batches to be used in the DataLoader. Defaults to 64.
            num_workers (int, optional): The number of worker processes to use for data loading. Defaults to 8.
            how (str, optional): The method to be used for selecting valid genes. Defaults to "most expr".
            max_len (int, optional): The maximum length of the gene sequence. Defaults to 1000.
            add_zero_genes (int, optional): The number of zero genes to add to the gene sequence. Defaults to 100.
            precision (str, optional): The precision to be used in the Trainer. Defaults to "16-mixed".
            organisms (List[str], optional): The list of organisms to be considered. Defaults to [ "NCBITaxon:9606", ].
            pred_embedding (List[str], optional): The list of labels to be used for plotting embeddings. Defaults to [ "cell_type_ontology_term_id", "disease_ontology_term_id", "self_reported_ethnicity_ontology_term_id", "sex_ontology_term_id", ].
            model_name (str, optional): The name of the model to be used. Defaults to "scprint".
            output_expression (str, optional): The type of output expression to be used. Can be one of "all", "sample", "none". Defaults to "sample".
        """
        self.model = model
        self.model.predict_mode = "denoise"
        self.model.predict_depth_mult = predict_depth_mult
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_len = max_len
        self.organisms = organisms
        self.plot_corr_size = plot_corr_size
        self.precision = precision
        self.doplot = doplot
        self.model.doplot = doplot
        self.trainer = Trainer(precision=precision)
        # subset_hvg=1000, use_layer='counts', is_symbol=True,force_preprocess=True, skip_validate=True)

    def __call__(self, adata: AnnData):
        random_indices = np.random.randint(
            low=0, high=adata.shape[0], size=self.plot_corr_size
        )
        if os.path.exists("collator_output.txt"):
            os.remove("collator_output.txt")
        adataset = SimpleAnnDataset(
            adata[random_indices], obs_to_output=["organism_ontology_term_id"]
        )
        adatac = sc.pp.log1p(adata, copy=True)
        sc.pp.highly_variable_genes(adatac)
        genelist = adata.var.index[
            np.argsort(adatac.var["dispersions_norm"].values)[::-1][: self.max_len]
        ].tolist()
        del adatac
        col = Collator(
            organisms=self.organisms,
            valid_genes=self.model.genes,
            how="some",
            genelist=genelist,
            downsample=0.4,
            save_output=True,
        )
        dataloader = DataLoader(
            adataset,
            collate_fn=col,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        self.trainer.predict(self.model, dataloader)

        reco = utils.zinb_sample(
            self.model.expr_pred[0],
            self.model.expr_pred[1],
            self.model.expr_pred[2],
        )
        noisy = np.loadtxt("collator_output.txt")
        true = adata.X[random_indices][
            :, adata.var.index.isin(self.model.genes) & adata.var.index.isin(genelist)
        ]

        corr_coef, p_value = spearmanr(
            np.vstack([reco.cpu().numpy(), noisy, true.todense()]).T
        )
        corr_coef[p_value > 0.05] = 0
        if self.doplot:
            plt.figure(figsize=(10, 5))
            plt.imshow(
                corr_coef, cmap="coolwarm", interpolation="none", vmin=-1, vmax=1
            )
            plt.colorbar()
            plt.title("Expression Correlation Coefficient")
            plt.show()
        metrics = {
            "reco_self_noisy": np.mean(
                corr_coef[
                    self.plot_corr_size : self.plot_corr_size * 2, : self.plot_corr_size
                ].diagonal()
            ),
            "reco_self_full": np.mean(
                corr_coef[self.plot_corr_size * 2 :, : self.plot_corr_size].diagonal()
            ),
            "reco_noisy_full": np.mean(
                corr_coef[
                    self.plot_corr_size * 2 :,
                    self.plot_corr_size : self.plot_corr_size * 2,
                ].diagonal()
            ),
        }
        return metrics
