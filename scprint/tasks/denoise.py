import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import torch

from scdataloader.data import SimpleAnnDataset
from scdataloader import Collator
from scprint.model import utils
import bionty as bt
from torch.utils.data import DataLoader
from typing import Tuple
import sklearn.metrics
import pandas as pd

from lightning.pytorch import Trainer
import anndata as ad

from typing import List
from anndata import AnnData
from scprint.tasks import compute_corr
from scdataloader import Preprocessor

from scipy.sparse import issparse

from scipy.stats import spearmanr
import os

from . import knn_smooth

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


class Denoiser:
    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int = 10,
        num_workers: int = 1,
        max_len: int = 20_000,
        precision: str = "16-mixed",
        organisms: List[str] = [
            "NCBITaxon:9606",
        ],
        plot_corr_size: int = 64,
        doplot: bool = True,
        predict_depth_mult: int = 4,
        downsample: float = 0.4,
        devices: List[int] = [0],
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
        self.downsample = downsample
        self.trainer = Trainer(precision=precision, devices=devices)
        # subset_hvg=1000, use_layer='counts', is_symbol=True,force_preprocess=True, skip_validate=True)

    def __call__(self, adata: AnnData):
        if os.path.exists("collator_output.txt"):
            os.remove("collator_output.txt")
        random_indices = None
        if self.plot_corr_size < adata.shape[0]:
            random_indices = np.random.randint(
                low=0, high=adata.shape[0], size=self.plot_corr_size
            )
            adataset = SimpleAnnDataset(
                adata[random_indices], obs_to_output=[
                    "organism_ontology_term_id"]
            )
        else:
            adataset = SimpleAnnDataset(
                adata, obs_to_output=["organism_ontology_term_id"]
            )
        sc.pp.highly_variable_genes(
            adata, flavor="seurat_v3", n_top_genes=self.max_len)
        genelist = adata.var.index[adata.var.highly_variable]
        print(len(genelist))
        col = Collator(
            organisms=self.organisms,
            valid_genes=self.model.genes,
            how="some",
            genelist=genelist,
            downsample=self.downsample,
            save_output=True,
        )
        dataloader = DataLoader(
            adataset,
            collate_fn=col,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        self.genes = list(set(self.model.genes) & set(genelist))

        self.model.doplot = self.doplot
        self.trainer.predict(self.model, dataloader)
        if self.downsample is not None:
            reco = self.model.expr_pred[0]
            noisy = np.loadtxt("collator_output.txt")
            true = adata.X[random_indices][
                :, adata.var.index.isin(self.genes)
            ] if random_indices is not None else adata.X[
                :, adata.var.index.isin(self.genes)
            ]

            true = true.toarray() if issparse(true) else true
            # noisy[true==0]=0
            reco = reco.cpu().numpy()
            # reco[true==0] = 0
            # import pdb
            # pdb.set_trace()
            # reco[reco!=0] = 2
            # corr_coef = np.corrcoef(
            #    np.vstack([reco[true!=0], noisy[true!=0], true[true!=0]])
            # )
            corr_coef, p_value = spearmanr(
                np.vstack(
                    [reco[true != 0], noisy[true != 0], true[true != 0]]).T
            )
            metrics = {
                "reco2noisy": corr_coef[0, 1],
                "reco2full": corr_coef[0, 2],
                "noisy2full": corr_coef[1, 2],
            }
            # corr_coef[p_value > 0.05] = 0
            # if self.doplot:
            #    plt.figure(figsize=(10, 5))
            #    plt.imshow(
            #        corr_coef, cmap="coolwarm", interpolation="none", vmin=-1, vmax=1
            #    )
            #    plt.colorbar()
            #    plt.title("Expression Correlation Coefficient")
            #    plt.show()
            # metrics = {
            #    "reco2noisy": np.mean(
            #        corr_coef[
            #            self.plot_corr_size : self.plot_corr_size * 2, : self.plot_corr_size
            #        ].diagonal()
            #    ),
            #    "reco2full": np.mean(
            #        corr_coef[self.plot_corr_size * 2 :, : self.plot_corr_size].diagonal()
            #    ),
            #    "noisy2full": np.mean(
            #        corr_coef[
            #            self.plot_corr_size * 2 :,
            #            self.plot_corr_size : self.plot_corr_size * 2,
            #        ].diagonal()
            #    ),
            # }
            return metrics, random_indices, self.genes, self.model.expr_pred[0]
        else:
            return random_indices, self.genes, self.model.expr_pred[0]

# testdatasets=['/R4ZHoQegxXdSFNFY5LGe.h5ad', '/SHV11AEetZOms4Wh7Ehb.h5ad',
# '/V6DPJx8rP3wWRQ43LMHb.h5ad', '/Gz5G2ETTEuuRDgwm7brA.h5ad', '/YyBdEsN89p2aF4xJY1CW.h5ad',
# '/SO5yBTUDBgkAmz0QbG8K.h5ad', '/r4iCehg3Tw5IbCLiCIbl.h5ad', '/SqvXr3i3PGXM8toXzUf9.h5ad',
# '/REIyQZE6OMZm1S3W2Dxi.h5ad', '/rYZ7gs0E0cqPOLONC8ia.h5ad', '/FcwMDDbAQPNYIjcYNxoc.h5ad',
# '/fvU5BAMJrm7vrgDmZM0z.h5ad', '/gNNpgpo6gATjuxTE7CCp.h5ad'],


def default_benchmark(model, default_dataset=FILE_DIR+"/../../data/r4iCehg3Tw5IbCLiCIbl.h5ad"):
    adata = sc.read_h5ad(default_dataset)
    denoise = Denoiser(
        model,
        batch_size=40,
        max_len=4000,
        plot_corr_size=480,
        doplot=False,
        predict_depth_mult=10,
        downsample=0.7,
        devices=1,
    )
    return denoise(adata)[0]


def open_benchmark(model):
    adata = sc.read(
        FILE_DIR+"/../../data/pancreas_atlas.h5ad",
        backup_url="https://figshare.com/ndownloader/files/24539828",
    )
    adata = adata[adata.obs.tech == "inDrop1"]

    train, test = split_molecules(
        adata.layers['counts'].round().astype(int), 0.9)
    is_missing = np.array(train.sum(axis=0) == 0)
    true = adata.copy()
    true.X = test
    adata.layers['counts'] = train
    test = test[:, ~is_missing.flatten()]
    adata = adata[:, ~is_missing.flatten()]

    adata.obs['organism_ontology_term_id'] = "NCBITaxon:9606"
    preprocessor = Preprocessor(subset_hvg=3000, use_layer='counts', is_symbol=True,
                                force_preprocess=True, skip_validate=True, do_postp=False)
    nadata = preprocessor(adata.copy())

    denoise = Denoiser(
        model,
        batch_size=32,
        max_len=15_800,
        plot_corr_size=10_000,
        doplot=False,
        predict_depth_mult=1.2,
        downsample=None,
    )
    expr = denoise(nadata)
    denoised = ad.AnnData(expr.cpu().numpy(), var=nadata.var.loc[np.array(
        denoise.model.genes)[denoise.model.pos[0].cpu().numpy().astype(int)]])
    denoised = denoised[:, denoised.var.symbol.isin(true.var.index)]
    loc = true.var.index.isin(denoised.var.symbol)
    true = true[:, loc]
    train = train[:, loc]

    # Ensure expr and adata are aligned by reordering expr to match adata's .var order
    denoised = denoised[:, denoised.var.set_index(
        'symbol').index.get_indexer(true.var.index)]
    denoised.X = np.maximum(denoised.X - train.astype(float), 0)
    # scaling and transformation
    target_sum = 1e4

    sc.pp.normalize_total(true, target_sum)
    sc.pp.log1p(true)

    sc.pp.normalize_total(denoised, target_sum)
    sc.pp.log1p(denoised)

    error_mse = sklearn.metrics.mean_squared_error(
        true.X, denoised.X
    )
    # scaling
    initial_sum = train.sum()
    target_sum = true.X.sum()
    denoised.X = denoised.X * target_sum / initial_sum
    error_poisson = poisson_nll_loss(true.X, denoised.X)
    return {"error_poisson": error_poisson, "error_mse": error_mse}


def mse(test_data, denoised_data, target_sum=1e4):
    sc.pp.normalize_total(test_data, target_sum)
    sc.pp.log1p(test_data)

    sc.pp.normalize_total(denoised_data, target_sum)
    sc.pp.log1p(denoised_data)

    print("Compute mse value", flush=True)
    return sklearn.metrics.mean_squared_error(
        test_data.X.todense(), denoised_data.X
    )


def withknn(adata, k=10, **kwargs):
    adata.layers['denoised'] = knn_smooth.knn_smoothing(
        adata.X.transpose(), k=k, **kwargs).transpose()
    return adata


# from molecular_cross_validation.mcv_sweep import poisson_nll_loss
# copied from: https://github.com/czbiohub/molecular-cross-validation/blob/master/src/molecular_cross_validation/mcv_sweep.py
def poisson_nll_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return (y_pred - y_true * np.log(y_pred + 1e-6)).mean()


def split_molecules(
    umis: np.ndarray,
    data_split: float,
    overlap_factor: float = 0.0,
    random_state: np.random.RandomState = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Splits molecules into two (potentially overlapping) groups.
    :param umis: Array of molecules to split
    :param data_split: Proportion of molecules to assign to the first group
    :param overlap_factor: Overlap correction factor, if desired
    :param random_state: For reproducible sampling
    :return: umis_X and umis_Y, representing ``split`` and ``~(1 - split)`` counts
             sampled from the input array
    """
    if random_state is None:
        random_state = np.random.RandomState()

    umis_X_disjoint = random_state.binomial(umis, data_split - overlap_factor)
    umis_Y_disjoint = random_state.binomial(
        umis - umis_X_disjoint, (1 - data_split) /
        (1 - data_split + overlap_factor)
    )
    overlap_factor = umis - umis_X_disjoint - umis_Y_disjoint
    umis_X = umis_X_disjoint + overlap_factor
    umis_Y = umis_Y_disjoint + overlap_factor

    return umis_X, umis_Y
