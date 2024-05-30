from sklearn.metrics import f1_score
from scdataloader import Preprocessor
from scdataloader.utils import get_descendants
from scib_metrics.benchmark import Benchmarker
from networkx import average_node_connectivity
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import torch

from scdataloader.data import SimpleAnnDataset
from scdataloader import Collator
from scprint.model import utils
import bionty as bt
from torch.utils.data import DataLoader
import os
import pandas as pd

from lightning.pytorch import Trainer

from scipy.stats import spearmanr

from typing import List
from anndata import AnnData

FILE_LOC = os.path.dirname(os.path.realpath(__file__))


class Embedder:
    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int = 64,
        num_workers: int = 8,
        how: str = "random expr",
        max_len: int = 2000,
        doclass: bool = True,
        add_zero_genes: int = 0,
        precision: str = "16-mixed",
        organisms: List[str] = [
            "NCBITaxon:9606",
            "NCBITaxon:10090",
        ],
        pred_embedding: List[str] = [
            "cell_type_ontology_term_id",
            "disease_ontology_term_id",
            "self_reported_ethnicity_ontology_term_id",
            "sex_ontology_term_id",
        ],
        model_name: str = "scprint",
        plot_corr_size: int = 64,
        doplot: bool = True,
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
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.how = how
        self.max_len = max_len
        self.add_zero_genes = add_zero_genes
        self.organisms = organisms
        self.pred_embedding = pred_embedding
        self.model_name = model_name
        self.plot_corr_size = plot_corr_size
        self.precision = precision
        self.doplot = doplot
        self.doclass = doclass
        self.model.doplot = doplot
        self.trainer = Trainer(precision=precision, devices=devices)
        # subset_hvg=1000, use_layer='counts', is_symbol=True,force_preprocess=True, skip_validate=True)

    def __call__(self, adata: AnnData, cache=False, output_expression: str = "none"):
        # one of "all" "sample" "none"
        try:
            mdir = (
                self.model.logger.save_dir
                if self.model.logger.save_dir is not None
                else "data"
            )
        except:
            mdir = "data"
        try:
            file = mdir + "/step_" + str(self.model.global_step) + "_predict_part_"+str(
                self.model.counter)+"_"+str(self.model.global_rank) + ".h5ad"
            hasfile = os.path.exists(file)
        except:
            hasfile = False
        if not cache or not hasfile:
            self.model.predict_mode = "none"
            # Add at least the organism you are working with
            adataset = SimpleAnnDataset(
                adata, obs_to_output=["organism_ontology_term_id"])
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
            self.model.pred_log_adata = True
            self.model.pred_embedding = self.pred_embedding

            self.trainer.predict(self.model, dataloader)
            try:
                mdir = (
                    self.model.logger.save_dir
                    if self.model.logger.save_dir is not None
                    else "data"
                )
            except:
                mdir = "data"
            file = mdir + "/step_" + str(self.model.global_step) + "_predict_part_"+str(
                self.model.counter)+"_"+str(self.model.global_rank) + ".h5ad"

        pred_adata = sc.read_h5ad(
            file
        )
        if output_expression == "all":
            adata.obsm["scprint_mu"] = self.model.expr_pred[0]
            adata.obsm["scprint_theta"] = self.model.expr_pred[1]
            adata.obsm["scprint_pi"] = self.model.expr_pred[2]
            adata.obsm["scprint_pos"] = self.model.pos.cpu().numpy()
        elif output_expression == "sample":
            adata.obsm["scprint_expr"] = utils.zinb_sample(
                self.model.expr_pred[0],
                self.model.expr_pred[1],
                self.model.expr_pred[2],
            ).cpu().numpy()
            adata.obsm["scprint_pos"] = self.model.pos.cpu().numpy()
        elif output_expression == "old":
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
            adata.obsm["scprint_pos"] = self.model.pos.cpu().numpy()
        else:
            pass
        pred_adata.obs.index = adata.obs.index
        adata.obsm["scprint_umap"] = pred_adata.obsm["X_umap"]
        # adata.obsm["scprint_leiden"] = pred_adata.obsm["leiden"]
        adata.obsm[self.model_name] = pred_adata.X
        pred_adata.obs.index = adata.obs.index
        adata.obs = pd.concat([adata.obs, pred_adata.obs], axis=1)
        metrics = {}
        if self.doclass:
            for cl in self.model.classes:
                res = []
                if cl not in adata.obs.columns:
                    continue
                class_topred = self.model.label_decoders[cl].values()

                if cl in self.model.labels_hierarchy:
                    # class_groupings = {
                    #    k: [
                    #        i.ontology_id
                    #        for i in bt.CellType.filter(k).first().children.all()
                    #    ]
                    #    for k in set(adata.obs[cl].unique()) - set(class_topred)
                    # }
                    cur_labels_hierarchy = {
                        self.model.label_decoders[cl][k]: [
                            self.model.label_decoders[cl][i] for i in v
                        ]
                        for k, v in self.model.labels_hierarchy[cl].items()
                    }
                else:
                    cur_labels_hierarchy = {}

                for pred, true in adata.obs[["pred_" + cl, cl]].values:
                    if pred == true:
                        res.append(True)
                        continue
                    if len(cur_labels_hierarchy) > 0:
                        if true in cur_labels_hierarchy:
                            res.append(pred in cur_labels_hierarchy[true])
                            continue
                        elif true not in class_topred:
                            raise ValueError(
                                f"true label {true} not in available classes")
                        elif true != "unknown":
                            res.append(False)
                    elif true not in class_topred:
                        raise ValueError(
                            f"true label {true} not in available classes")
                    elif true != "unknown":
                        res.append(False)
                    # else true is unknown
                    # else we pass
                if len(res) == 0:
                    # true was always unknown
                    res = [1]
                if self.doplot:
                    print("    ", cl)
                    print("     accuracy:", sum(res) / len(res))
                    print(" ")
                metrics.update({cl + "_accuracy": sum(res) / len(res)})
        # m = self.compute_reconstruction(adata, plot_corr_size=self.plot_corr_size)
        # metrics.update(m)
        return adata, metrics

    def compute_reconstruction(self, adata, plot_corr_size=64):
        if plot_corr_size < 1:
            raise ValueError("plot_corr_size should be greater than 0")
        sc.pp.highly_variable_genes(
            adata, flavor="seurat_v3", n_top_genes=self.max_len)
        highly_variable = adata.var.index[adata.var.highly_variable]
        random_indices = np.random.randint(
            low=0, high=adata.shape[0], size=plot_corr_size
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
            batch_size=plot_corr_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        self.model.pred_log_adata = False
        self.model.predict_mode = "generate"

        # self.trainer.num_predict_batches = 1

        self.trainer.predict(self.model, dataloader)

        res = self.model.expr_pred
        # pos = adata.obsm["scprint_pos"][random_indices]
        if len(res) > 1:
            out = (
                utils.zinb_sample(
                    res[0],
                    res[1],
                    res[2],
                )
                .cpu()
                .numpy()
            )
            try:
                mean_expr = pd.read_parquet("../../data/avg_expr.parquet")
                genes_used = [self.model.genes[int(i)]
                              for i in self.model.pos[0]]
                mean_expr = mean_expr[mean_expr.index.isin(genes_used)][
                    ["avg_expr", "avg_expr_wexpr"]
                ].values
                out = np.hstack([out.T, mean_expr])
                add = 2
            except:
                print(
                    "cannot read the mean expr file under scprint/data/avg_expr.parquet")
                out = out.T
                mean_expr = None
                add = 0

            to = adata[
                random_indices,
                adata.var.index.isin(set(highly_variable)
                                     & set(self.model.genes)),
            ].X.todense()
            metrics = compute_corr(
                out,
                to,
                doplot=self.doplot,
                compute_mean_regress=add == 2,
                plot_corr_size=plot_corr_size,
            )
        expr = res[0].cpu().numpy()
        # expr[
        #    np.random.binomial(
        #        1,
        #        p=torch.nn.functional.sigmoid(res[2].to(torch.float32)).cpu().numpy(),
        #    ).astype(bool)
        # ] = 0
        compute_corr(expr.T, to, doplot=self.doplot,
                     plot_corr_size=plot_corr_size)
        return metrics


def compute_corr(out, to, doplot=True, compute_mean_regress=False, plot_corr_size=64):
    metrics = {}
    corr_coef, p_value = spearmanr(
        out,
        to.T,
    )
    corr_coef[p_value > 0.05] = 0
    # corr_coef[]
    # only on non zero values,
    # compare a1-b1 corr with a1-b(n) corr. should be higher

    # Plot correlation coefficient
    val = plot_corr_size + 2 if compute_mean_regress else plot_corr_size
    metrics.update(
        {"recons_corr": np.mean(corr_coef[val:, :plot_corr_size].diagonal())}
    )
    if compute_mean_regress:
        metrics.update(
            {
                "mean_regress": np.mean(
                    corr_coef[
                        plot_corr_size: plot_corr_size + 2,
                        :plot_corr_size,
                    ].flatten()
                )
            }
        )
    if doplot:
        plt.figure(figsize=(10, 5))
        plt.imshow(corr_coef, cmap="coolwarm",
                   interpolation="none", vmin=-1, vmax=1)
        plt.colorbar()
        plt.title('Correlation Coefficient of expr and i["x"]')
        plt.show()
    return metrics


def default_benchmark(model, default_dataset="pancreas", do_class=True, coarse=False):
    if default_dataset == 'pancreas':
        adata = sc.read(
            FILE_LOC+"/../../data/pancreas_atlas.h5ad",
            backup_url="https://figshare.com/ndownloader/files/24539828",
        )
        adata.obs['cell_type_ontology_term_id'] = adata.obs['celltype'].replace(
            COARSE if coarse else FINE)
        adata.obs['assay_ontology_term_id'] = adata.obs['tech'].replace(
            COARSE if coarse else FINE)
    elif default_dataset == 'lung':
        adata = sc.read(
            FILE_LOC+"/../../data/lung_atlas.h5ad",
            backup_url="https://figshare.com/ndownloader/files/24539942",
        )
        adata.obs['cell_type_ontology_term_id'] = adata.obs['cell_type'].replace(
            COARSE if coarse else FINE)
    else:
        adata = sc.read_h5ad(default_dataset)
        adata.obs['batch'] = adata.obs['assay_ontology_term_id']
        adata.obs['cell_type'] = adata.obs['cell_type_ontology_term_id']
    preprocessor = Preprocessor(
        use_layer="counts",
        is_symbol=True,
        force_preprocess=True,
        skip_validate=True,
        do_postp=False,
    )
    adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    adata = preprocessor(adata.copy())
    embedder = Embedder(
        model, pred_embedding=["cell_type_ontology_term_id"], organisms=[adata.obs["organism_ontology_term_id"].values[0]],
        doclass=(default_dataset not in ['pancreas', 'lung']), devices=1,
    )
    embed_adata, metrics = embedder(adata.copy())

    bm = Benchmarker(
        embed_adata,
        batch_key="tech" if default_dataset == 'pancreas' else "batch",
        label_key="celltype" if default_dataset == 'pancreas' else "cell_type",
        embedding_obsm_keys=["scprint"],
        n_jobs=6,
    )
    bm.benchmark()
    metrics.update({"scib": bm.get_results(
        min_max_scale=False).T.to_dict()["scprint"]})
    metrics['classif'] = compute_classification(embed_adata, model.classes,
                                                model.label_decoders, model.labels_hierarchy)
    return metrics


def compute_classification(adata, classes, label_decoders, labels_hierarchy,
                           metric_type=["macro", "micro", "weighted"]):
    metrics = {}
    for label in classes:
        res = []
        if label not in adata.obs.columns:
            continue
        labels_topred = label_decoders[label].values()
        if label in labels_hierarchy:
            parentdf = (
                bt.CellType.filter()
                .df(include=["parents__ontology_id"])
                .set_index("ontology_id")[['parents__ontology_id']]
            )
            parentdf.parents__ontology_id = parentdf.parents__ontology_id.astype(
                str)
            class_groupings = {
                k: get_descendants(k, parentdf)
                for k in set(adata.obs[label].unique())
            }
        for pred, true in adata.obs[["pred_" + label, label]].values:
            if pred == true:
                res.append(true)
                continue
            if label in labels_hierarchy:
                if true in class_groupings:
                    res.append(true if pred in class_groupings[true] else "")
                    continue
                elif true not in labels_topred:
                    raise ValueError(
                        f"true label {true} not in available classes")
            elif true not in labels_topred:
                raise ValueError(f"true label {true} not in available classes")
            res.append("")
        metrics[label] = {}
        metrics[label]['accuracy'] = np.mean(
            np.array(res) == adata.obs[label].values)
        for x in metric_type:
            metrics[label][x] = f1_score(
                np.array(res),
                adata.obs[label].values,
                average=x
            )
    return metrics


FINE = {"gamma": "CL:0002275",
        "beta": "CL:0000169",  # "CL:0008024"
        "epsilon": "CL:0005019",  # "CL:0008024"
        "acinar": "CL:0000622",
        "delta": "CL:0000173",  # "CL:0008024"
        "schwann": "CL:0002573",  # "CL:0000125"
        "activated_stellate": "CL:0000057",
        "alpha": "CL:0000171",  # "CL:0008024"
        "mast": "CL:0000097",
        "Mast cell": "CL:0000097",
        "quiescent_stellate": "CL:0000057",
        "t_cell": "CL:0000084",
        "endothelial": "CL:0000115",
        "Endothelium": "CL:0000115",
        "ductal": "CL:0002079",  # CL:0000068
        "macrophage": "CL:0000235",
        "Macrophage": "CL:0000235",
        "B cell": "CL:0000236",
        "Type 2": "CL:0002063",
        "Type 1": "CL:0002062",
        "Ciliated": "CL:4030034",  # respiratory ciliated
        "Dendritic cell": "CL:0000451",  # leukocyte
        "Secretory": "CL:0000151",
        "Ionocytes": "CL:0005006",
        "Basal 1": "CL:0000646",  # epithelial
        "Basal 2": "CL:0000646",
        "Secretory": "CL:0000151",
        "Neutrophil_CD14_high": "CL:0000775",
        "Neutrophils_IL1R2": "CL:0000775",
        "Lymphatic": "CL:0002138",
        "Fibroblast": "CL:0000057",
        "T/NK cell": "CL:0000814",

        "inDrop1": "EFO:0008780",
        "inDrop3": "EFO:0008780",
        "inDrop4": "EFO:0008780",
        "inDrop2": "EFO:0008780",
        "fluidigmc1": "EFO:0010058",  # fluidigm c1
        "smarter": "EFO:0010058",  # fluidigm c1
        "celseq2": "EFO:0010010",
        "smartseq2": "EFO:0008931",
        "celseq": "EFO:0008679",
        }
COARSE = {
    "beta": "CL:0008024",  # endocrine
    "epsilon": "CL:0008024",
    "delta": "CL:0008024",
    "alpha": "CL:0008024",
    "gamma": "CL:0008024",

    "acinar": "CL:0000150",  # epithelial (gland)
    "ductal": "CL:0000068",  # epithelial (duct)

    "schwann": "CL:0000125",  # glial

    "endothelial": "CL:0000115",
    "Endothelium": "CL:0000115",
    "Lymphatic": "CL:0000115",

    "macrophage": "CL:0000235",  # myeloid leukocyte (not)
    "Macrophage": "CL:0000235",  # myeloid leukocyte
    "mast": "CL:0000097",  # myeloid leukocyte (not)
    "Mast cell": "CL:0000097",  # myeloid leukocyte
    "Neutrophil_CD14_high": "CL:0000775",  # myeloid leukocyte
    "Neutrophils_IL1R2": "CL:0000775",  # myeloid leukocyte

    "t_cell": "CL:0000084",  # leukocyte, lymphocyte (not)
    "T/NK cell": "CL:0000084",  # leukocyte, lymphocyte (not)
    "B cell": "CL:0000236",  # leukocyte, lymphocyte (not)
    "Dendritic cell": "CL:0000451",  # leukocyte, lymphocyte

    "activated_stellate": "CL:0000057",  # fibroblast (not)
    "quiescent_stellate": "CL:0000057",  # fibroblast (not)
    "Fibroblast": "CL:0000057",

    "Type 2": "CL:0000066",  # epithelial
    "Type 1": "CL:0000066",
    "Ionocytes": "CL:0000066",  # epithelial
    "Basal 1": "CL:0000066",  # epithelial
    "Basal 2": "CL:0000066",

    "Ciliated": "CL:0000064",  # ciliated

    "Secretory": "CL:0000151",

    "inDrop1": "EFO:0008780",
    "inDrop3": "EFO:0008780",
    "inDrop4": "EFO:0008780",
    "inDrop2": "EFO:0008780",
    'fluidigmc1': 'EFO:0010058',  # fluidigm c1
    'smarter': 'EFO:0010058',  # fluidigm c1
    'celseq2': 'EFO:0010010',
    'smartseq2': 'EFO:0008931',
    'celseq': 'EFO:0008679',
}
