from bengrn import BenGRN, get_sroy_gt, get_perturb_gt
from scdataloader import Preprocessor
from bengrn.base import train_classifier
from bengrn import BenGRN, get_sroy_gt
from grnndata import utils as grnutils
from anndata.utils import make_index_unique
import scanpy as sc
import torch
import gc
from scdataloader.data import SimpleAnnDataset
from scdataloader import Collator
from grnndata import utils
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer
import joblib
from typing import List
from anndata import AnnData

from scprint.utils.sinkhorn import SinkhornDistance
from scprint.utils import load_genes

from grnndata import GRNAnnData, from_anndata, read_h5ad

import umap
import hdbscan
from tqdm import tqdm

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from .tmfg import tmfg
import networkx as nx
import scipy.sparse
import os.path

import pandas as pd

FILEDIR = os.path.dirname(os.path.realpath(__file__))


class GRNfer:
    def __init__(
        self,
        model: torch.nn.Module,
        adata: AnnData,
        batch_size: int = 64,
        num_workers: int = 8,
        drop_unexpressed: bool = False,
        num_genes: int = 3000,
        precision: str = "16-mixed",
        cell_type_col="cell_type",
        model_name: str = "scprint",
        how: str = "random expr",  # random expr, most var withing, most var across, given
        preprocess="softmax",  # sinkhorn, softmax, none
        head_agg="mean",  # mean, sum, none
        filtration="thresh",  # thresh, top-k, mst, known, none
        k=10,
        apc=False,
        known_grn=None,
        symmetrize=False,
        doplot=True,
        max_cells=0,
        forward_mode="none",
        genes: list = [],
        loc="./",
        dtype=torch.float16,
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
            pred_embedding (List[str], optional): The list of labels to be used for plotting embeddings. Defaults to [ "cell_type_ontology_term_id", "disease_ontology_term_id", "self_reported_ethnicity_ontology_term_id", "sex_ontology_term_id", ].
            model_name (str, optional): The name of the model to be used. Defaults to "scprint".
            output_expression (str, optional): The type of output expression to be used. Can be one of "all", "sample", "none". Defaults to "sample".
        """
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.how = how
        assert self.how in [
            "most var within",
            "most var across",
            "random expr",
            "given",
        ], "how must be one of 'most var within', 'most var across', 'random expr', 'given'"
        self.num_genes = num_genes
        self.model_name = model_name
        self.adata = adata
        self.preprocess = preprocess
        self.cell_type_col = cell_type_col
        self.filtration = filtration
        self.doplot = doplot
        self.genes = genes
        self.apc = apc
        self.forward_mode = forward_mode
        self.k = k
        self.symmetrize = symmetrize
        self.known_grn = known_grn
        self.head_agg = head_agg
        self.max_cells = max_cells
        self.curr_genes = None
        self.drop_unexpressed = drop_unexpressed
        self.precision = precision
        ##elf.trainer = Trainer(precision=precision, devices=devices, use_distributed_sampler=False)
        # subset_hvg=1000, use_layer='counts', is_symbol=True,force_preprocess=True, skip_validate=True)

    def __call__(self, layer, cell_type=None, locname=""):
        # Add at least the organism you are working with
        subadata = self.predict(layer, cell_type)
        adjacencies = self.aggregate(self.model.attn.get())
        if self.head_agg == "none":
            return self.save(adjacencies[8:, 8:, :], subadata, locname)
        else:
            return self.save(self.filter(adjacencies)[8:, 8:], subadata, locname)

    def predict(self, layer, cell_type=None):
        self.curr_genes = None
        self.model.pred_log_adata = False
        if cell_type is not None:
            subadata = self.adata[
                self.adata.obs[self.cell_type_col] == cell_type
            ].copy()
        else:
            subadata = self.adata.copy()
        if self.how == "most var within":
            sc.pp.highly_variable_genes(
                subadata, flavor="seurat_v3", n_top_genes=self.num_genes
            )
            self.curr_genes = (
                subadata.var.index[subadata.var.highly_variable].tolist() + self.genes
            )
            print(
                "number of expressed genes in this cell type: "
                + str((subadata.X.sum(0) > 1).sum())
            )
        elif self.how == "most var across" and cell_type is not None:
            sc.tl.rank_genes_groups(
                self.adata,
                mask_var=self.adata.var.index.isin(self.model.genes),
                groupby=self.cell_type_col,
                groups=[cell_type],
            )
            diff_expr_genes = self.adata.uns["rank_genes_groups"]["names"][cell_type]
            diff_expr_genes = [
                gene for gene in diff_expr_genes if gene in self.model.genes
            ]
            self.curr_genes = diff_expr_genes[:self.num_genes] + self.genes
            self.curr_genes.sort()
        elif self.how == "random expr":
            self.curr_genes = self.model.genes
            # raise ValueError("cannot do it yet")
            pass
        elif self.how == "given" and len(self.genes) > 0:
            self.curr_genes = self.genes
        else:
            raise ValueError("how must be one of 'most var', 'random expr'")
        if self.drop_unexpressed:
            expr = subadata.var[(subadata.X.sum(0) > 0).tolist()[0]].index.tolist()
            self.curr_genes = [i for i in self.curr_genes if i in expr]
        subadata = subadata[: self.max_cells] if self.max_cells else subadata
        if len(subadata) == 0:
            raise ValueError("no cells in the dataset")
        adataset = SimpleAnnDataset(
            subadata, obs_to_output=["organism_ontology_term_id"]
        )
        self.col = Collator(
            organisms=self.model.organisms,
            valid_genes=self.model.genes,
            how="some" if self.how != "random expr" else "random expr",
            genelist=self.curr_genes if self.how != "random expr" else [],
        )
        dataloader = DataLoader(
            adataset,
            collate_fn=self.col,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        self.model.attn.comp_attn = self.head_agg == "mean_full"
        self.model.doplot = self.doplot
        self.model.on_predict_epoch_start()
        self.model.eval()
        device = self.model.device.type

        with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
            for batch in tqdm(dataloader):
                gene_pos, expression, depth = (
                    batch["genes"].to(device),
                    batch["x"].to(device),
                    batch["depth"].to(device),
                )
                self.model._predict(
                    gene_pos,
                    expression,
                    depth,
                    predict_mode=self.forward_mode,
                    get_attention_layer=layer if type(layer) is list else [layer],
                )
                torch.cuda.empty_cache()
        return subadata

    def aggregate(self, attn):
        if self.head_agg == "mean_full":
            self.curr_genes = [i for i in self.model.genes if i in self.curr_genes]
            return attn
        badloc = torch.isnan(attn.sum((0, 2, 3, 4)))
        attn = attn[:, ~badloc, :, :, :]
        self.curr_genes = (
            np.array(self.curr_genes)[~badloc[8:]]
            if self.how == "random expr"
            else [i for i in self.model.genes if i in self.curr_genes]
        )
        if self.doplot:
            sns.set_theme(
                style="white", context="poster", rc={"figure.figsize": (14, 10)}
            )
            fit = umap.UMAP()
            mm = fit.fit_transform(attn[0, :, 0, 0, :].detach().cpu().numpy())
            labels = hdbscan.HDBSCAN(
                min_samples=10,
                min_cluster_size=100,
            ).fit_predict(mm)
            plt.scatter(mm[:, 0], mm[:, 1], c=labels)
            plt.title(f"Qs @H{0}")
            plt.show()
            mm = fit.fit_transform(attn[0, :, 1, 0, :].detach().cpu().numpy())
            labels = hdbscan.HDBSCAN(
                min_samples=10,
                min_cluster_size=100,
            ).fit_predict(mm)
            plt.scatter(mm[:, 0], mm[:, 1], c=labels)
            plt.title(f"Ks @H{0}")
            plt.show()
        # attn = attn[:, :, 0, :, :].permute(0, 2, 1, 3) @ attn[:, :, 1, :, :].permute(
        #    0, 2, 3, 1
        # )
        attns = None
        Qs = (
            attn[:, :, 0, :, :]
            .permute(0, 2, 1, 3)
            .reshape(-1, attn.shape[1], attn.shape[-1])
        )
        Ks = (
            attn[:, :, 1, :, :]
            .permute(0, 2, 1, 3)
            .reshape(-1, attn.shape[1], attn.shape[-1])
        )
        for i in range(Qs.shape[0]):
            attn = Qs[i] @ Ks[i].T
            # return attn
            scale = Qs.shape[-1] ** -0.5
            attn = attn * scale
            if self.preprocess == "sinkhorn":
                if attn.numel() > 100_000_000:
                    raise ValueError("you can't sinkhorn such a large matrix")
                sink = SinkhornDistance(0.1, max_iter=200)
                attn = sink(attn)[0]
                attn = attn * Qs.shape[-1]
            elif self.preprocess == "softmax":
                attn = torch.nn.functional.softmax(attn, dim=-1)
            elif self.preprocess == "none":
                pass
            else:
                raise ValueError(
                    "preprocess must be one of 'sinkhorn', 'softmax', 'none'"
                )

            if self.symmetrize:
                attn = (attn + attn.T) / 2
            if self.apc:
                pass
                # attn = attn - (
                #    (attn.sum(-1).unsqueeze(-1) * attn.sum(-2).unsqueeze(-2))
                #    / attn.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
                # )  # .view()
            if self.head_agg == "mean":
                attns = attn.detach().cpu().numpy() + (
                    attns if attns is not None else 0
                )
            elif self.head_agg == "max":
                attns = (
                    np.maximum(attn.detach().cpu().numpy(), attns)
                    if attns is not None
                    else attn.detach().cpu().numpy()
                )
            elif self.head_agg == "none":
                if attns is not None:
                    if len(attns.shape) > 2:
                        attns = np.concatenate(
                            [attns, attn.detach().cpu().numpy()[..., np.newaxis]],
                            axis=-1,
                        )
                    else:
                        attns = np.stack([attns, attn.detach().cpu().numpy()], axis=-1)

                else:
                    attns = attn.detach().cpu().numpy()
            else:
                raise ValueError("head_agg must be one of 'mean', 'max' or 'None'")
        if self.head_agg == "mean":
            attns = attns / Qs.shape[0]
        return attns

    def filter(self, adj, gt=None):
        if self.filtration == "thresh":
            adj[adj < (1 / adj.shape[-1])] = 0
            # res = (adj != 0).sum()
            # if res / adj.shape[0] ** 2 < 0.01:
            #   adj = scipy.sparse.csr_matrix(adj)
        elif self.filtration == "none":
            pass
        elif self.filtration == "top-k":
            args = np.argsort(adj)
            adj[np.arange(adj.shape[0])[:, None], args[:, : -self.k]] = 0
            adj = scipy.sparse.csr_matrix(adj)
        elif self.filtration == "known" and gt is not None:
            gt = gt.reindex(sorted(gt.columns), axis=1)
            gt = gt.reindex(sorted(gt.columns), axis=0)
            gt = gt[gt.index.isin(self.curr_genes)].iloc[
                :, gt.columns.isin(self.curr_genes)
            ]

            loc = np.isin(self.curr_genes, gt.index)
            self.curr_genes = np.array(self.curr_genes)[loc]
            adj = adj[8:, 8:][loc][:, loc]
            adj[gt.values != 1] = 0
            adj = scipy.sparse.csr_matrix(adj)
        elif self.filtration == "tmfg":
            adj = nx.to_scipy_sparse_array(tmfg(adj))
        elif self.filtration == "mst":
            pass
        else:
            raise ValueError("filtration must be one of 'thresh', 'none' or 'top-k'")
        res = (adj != 0).sum() if self.filtration != "none" else adj.shape[0] ** 2
        print(f"avg link count: {res}, sparsity: {res / adj.shape[0] ** 2}")
        return adj

    def save(self, grn, subadata, loc=""):
        grn = GRNAnnData(
            subadata[:, subadata.var.index.isin(self.curr_genes)].copy(),
            grn=grn,
        )
        # grn = grn[:, (grn.X != 0).sum(0) > (self.max_cells / 32)]
        grn.var["TFs"] = [True if i in utils.TF else False for i in grn.var["symbol"]]
        grn.uns["grn_scprint_params"] = {
            "filtration": self.filtration,
            "how": self.how,
            "preprocess": self.preprocess,
            "head_agg": self.head_agg,
        }
        if loc != "":
            grn.write_h5ad(loc + "grn_fromscprint.h5ad")
            return from_anndata(grn)
        else:
            return grn


def get_GTdb(db="omnipath"):
    if db == "omnipath":
        if not os.path.exists(FILEDIR + "/../../data/main/omnipath.parquet"):
            from omnipath.interactions import AllInteractions
            from omnipath.requests import Annotations

            interactions = AllInteractions()
            net = interactions.get(exclude=["small_molecule", "lncrna_mrna"])
            hgnc = Annotations.get(resources="HGNC")
            rename = {v.uniprot: v.genesymbol for _, v in hgnc.iterrows()}
            net = net.replace({"source": rename, "target": rename})
            genedf = load_genes()
            rn = {
                j["symbol"]: i
                for i, j in genedf[["symbol"]].iterrows()
                if j["symbol"] is not None
            }
            net = net.replace({"source": rn, "target": rn})
            varnames = list(set(net.iloc[:, :2].values.flatten()))
            da = np.zeros((len(varnames), len(varnames)), dtype=float)
            for i, j in net.iloc[:, :2].values:
                da[varnames.index(i), varnames.index(j)] = 1
            net = pd.DataFrame(data=da, index=varnames, columns=varnames)
            net.to_parquet(FILEDIR + "/../../data/main/omnipath.parquet")
        else:
            net = pd.read_parquet(FILEDIR + "/../../data/main/omnipath.parquet")
    if db == "scenic+":
        net = pd.read_parquet(FILEDIR + "/../../data/main/main_scenic+.parquet")
    if db == "stringdb":
        net = pd.read_parquet(FILEDIR + "/../../data/main/stringdb_bias.parquet")
    return net


def default_benchmark(
    model,
    default_dataset="sroy",
    cell_types=[
        "kidney distal convoluted tubule epithelial cell",
        "kidney loop of Henle thick ascending limb epithelial cell",
        "kidney collecting duct principal cell",
        "mesangial cell",
        "blood vessel smooth muscle cell",
        "podocyte",
        "macrophage",
        "leukocyte",
        "kidney interstitial fibroblast",
        "endothelial cell",
    ],
    maxlayers=16,
    maxgenes=5000,
    batch_size=32,
    maxcells=1024,
):
    metrics = {}
    layers = list(range(model.nlayers))[max(0, model.nlayers - maxlayers) :]
    clf_omni = None
    if default_dataset == "sroy":
        preprocessor = Preprocessor(
            is_symbol=True,
            force_preprocess=True,
            skip_validate=True,
            do_postp=False,
            min_valid_genes_id=5000,
            min_dataset_size=64,
        )
        clf_self = None
        todo = [
            ("han", "human", "full"),
            ("mine", "human", "full"),
            ("han", "human", "chip"),
            ("han", "human", "ko"),
            ("tran", "mouse", "full"),
            ("zhao", "mouse", "full"),
            ("tran", "mouse", "chip"),
            ("tran", "mouse", "ko"),
        ]
        for da, spe, gt in todo:
            if gt != "full":
                continue
            print(da + "_" + gt)
            preadata = get_sroy_gt(get=da, species=spe, gt=gt)
            adata = preprocessor(preadata.copy())
            grn_inferer = GRNfer(
                model,
                adata,
                how="most var within",
                preprocess="softmax",
                head_agg="none",
                filtration="none",
                forward_mode="none",
                num_genes=maxgenes,
                num_workers=8,
                max_cells=maxcells,
                doplot=False,
                batch_size=batch_size,
                devices=1,
            )
            grn = grn_inferer(layer=layers)
            grn.varp["all"] = grn.varp["GRN"]
            grn.var["ensembl_id"] = grn.var.index
            grn.var["symbol"] = make_index_unique(grn.var["symbol"].astype(str))
            grn.var.index = grn.var["symbol"]
            grn.varp["GRN"] = grn.varp["all"].mean(-1).T
            metrics["mean_" + da + "_" + gt] = BenGRN(
                grn, do_auc=True, doplot=False
            ).compare_to(other=preadata)
            grn.varp["GRN"] = grn.varp["GRN"].T
            if spe == "human":
                metrics["mean_" + da + "_" + gt + "_base"] = BenGRN(
                    grn, do_auc=True, doplot=False
                ).scprint_benchmark()

            ## OMNI
            if clf_omni is None:
                grn.varp["GRN"] = grn.varp["all"]
                _, m, clf_omni = train_classifier(
                    grn,
                    C=1,
                    train_size=0.9,
                    class_weight={1: 800, 0: 1},
                    shuffle=True,
                    return_full=False,
                )
                joblib.dump(clf_omni, "clf_omni.pkl")
                metrics["omni_classifier"] = m
            grn.varp["GRN"] = grn.varp["all"][:, :, clf_omni.coef_[0] > 0].mean(-1)
            if spe == "human":
                metrics["omni_" + da + "_" + gt + "_base"] = BenGRN(
                    grn, do_auc=True, doplot=True
                ).scprint_benchmark()
            grn.varp["GRN"] = grn.varp["GRN"].T
            metrics["omni_" + da + "_" + gt] = BenGRN(
                grn, do_auc=True, doplot=False
            ).compare_to(other=preadata)

            ## SELF
            if clf_self is None:
                grn.varp["GRN"] = np.transpose(grn.varp["all"], (1, 0, 2))
                _, m, clf_self = train_classifier(
                    grn,
                    other=preadata,
                    C=1,
                    train_size=0.5,
                    class_weight={1: 40, 0: 1},
                    shuffle=False,
                    return_full=False,
                )
                metrics["self_classifier"] = m
            grn.varp["GRN"] = grn.varp["all"][:, :, clf_self.coef_[0] > 0].mean(-1).T
            metrics["self_" + da + "_" + gt] = BenGRN(
                grn, do_auc=True, doplot=False
            ).compare_to(other=preadata)
            if spe == "human":
                grn.varp["GRN"] = grn.varp["GRN"].T
                metrics["self_" + da + "_" + gt + "_base"] = BenGRN(
                    grn, do_auc=True, doplot=True
                ).scprint_benchmark()

            ## chip / ko
            if (da, spe, "chip") in todo:
                preadata = get_sroy_gt(get=da, species=spe, gt="chip")
                grn.varp["GRN"] = grn.varp["all"].mean(-1).T
                metrics["mean_" + da + "_" + "chip"] = BenGRN(
                    grn, do_auc=True, doplot=False
                ).compare_to(other=preadata)
                grn.varp["GRN"] = (
                    grn.varp["all"][:, :, clf_omni.coef_[0] > 0].mean(-1).T
                )
                metrics["omni_" + da + "_" + "chip"] = BenGRN(
                    grn, do_auc=True, doplot=False
                ).compare_to(other=preadata)
                grn.varp["GRN"] = (
                    grn.varp["all"][:, :, clf_self.coef_[0] > 0].mean(-1).T
                )
                metrics["self_" + da + "_" + "chip"] = BenGRN(
                    grn, do_auc=True, doplot=False
                ).compare_to(other=preadata)
            if (da, spe, "ko") in todo:
                preadata = get_sroy_gt(get=da, species=spe, gt="ko")
                grn.varp["GRN"] = grn.varp["all"].mean(-1).T
                metrics["mean_" + da + "_" + "ko"] = BenGRN(
                    grn, do_auc=True, doplot=False
                ).compare_to(other=preadata)
                grn.varp["GRN"] = (
                    grn.varp["all"][:, :, clf_omni.coef_[0] > 0].mean(-1).T
                )
                metrics["omni_" + da + "_" + "ko"] = BenGRN(
                    grn, do_auc=True, doplot=False
                ).compare_to(other=preadata)
                grn.varp["GRN"] = (
                    grn.varp["all"][:, :, clf_self.coef_[0] > 0].mean(-1).T
                )
                metrics["self_" + da + "_" + "ko"] = BenGRN(
                    grn, do_auc=True, doplot=False
                ).compare_to(other=preadata)
            del grn
    elif default_dataset == "gwps":
        if not os.path.exists(FILEDIR + "/../../data/perturb_gt.h5ad"):
            adata = get_perturb_gt()
            adata.write_h5ad(FILEDIR + "/../../data/perturb_gt.h5ad")
        else:
            adata = read_h5ad(FILEDIR + "/../../data/perturb_gt.h5ad")
        preprocessor = Preprocessor(
            force_preprocess=True,
            skip_validate=True,
            do_postp=False,
            min_valid_genes_id=maxgenes,
            min_dataset_size=64,
        )
        nadata = preprocessor(adata.copy())
        adata.var["isTF"] = False
        adata.var.loc[adata.var.gene_name.isin(grnutils.TF), "isTF"] = True
        adata.var["isTF"].sum()
        grn_inferer = GRNfer(
            model,
            nadata,
            how="most var within",
            preprocess="softmax",
            head_agg="none",
            filtration="none",
            forward_mode="none",
            num_genes=maxgenes,
            max_cells=maxcells,
            doplot=False,
            num_workers=8,
            batch_size=batch_size,
            devices=1,
        )
        grn = grn_inferer(layer=layers)
        grn.varp["all"] = grn.varp["GRN"]

        grn.varp["GRN"] = grn.varp["all"].mean(-1).T
        metrics["mean"] = BenGRN(grn, do_auc=True, doplot=False).compare_to(other=adata)
        grn.var["ensembl_id"] = grn.var.index
        grn.var.index = grn.var["symbol"]
        grn.varp["GRN"] = grn.varp["all"].mean(-1)
        metrics["mean_base"] = BenGRN(
            grn, do_auc=True, doplot=False
        ).scprint_benchmark()

        grn.varp["GRN"] = grn.varp["all"]
        grn.var.index = grn.var["ensembl_id"]
        _, m, clf_omni = train_classifier(
            grn,
            C=1,
            train_size=0.9,
            class_weight={1: 800, 0: 1},
            shuffle=True,
            doplot=False,
            return_full=False,
            use_col="gene_name",
        )
        grn.varp["GRN"] = grn.varp["all"][:, :, clf_omni.coef_[0] > 0].mean(-1).T
        metrics["omni"] = BenGRN(grn, do_auc=True, doplot=False).compare_to(other=adata)
        metrics["omni_classifier"] = m
        grn.var.index = grn.var["symbol"]
        grn.varp["GRN"] = grn.varp["GRN"].T
        metrics["omni_base"] = BenGRN(
            grn, do_auc=True, doplot=False
        ).scprint_benchmark()
        grn.varp["GRN"] = np.transpose(grn.varp["all"], (1, 0, 2))
        grn.var.index = grn.var["ensembl_id"]
        _, m, clf_self = train_classifier(
            grn,
            other=adata,
            C=1,
            train_size=0.5,
            class_weight={1: 40, 0: 1},
            doplot=False,
            shuffle=False,
            return_full=False,
            use_col="ensembl_id",
        )
        grn.varp["GRN"] = grn.varp["all"][:, :, clf_self.coef_[0] > 0].mean(-1).T
        metrics["self"] = BenGRN(grn, do_auc=True, doplot=False).compare_to(other=adata)
        metrics["self_classifier"] = m
        grn.var.index = grn.var["symbol"]
        grn.varp["GRN"] = grn.varp["GRN"].T
        metrics["self_base"] = BenGRN(
            grn, do_auc=True, doplot=False
        ).scprint_benchmark()
    else:
        # max_genes=4000
        adata = sc.read_h5ad(default_dataset)
        adata.var["isTF"] = False
        adata.var.loc[adata.var.symbol.isin(grnutils.TF), "isTF"] = True
        for celltype in cell_types:
            print(celltype)
            grn_inferer = GRNfer(
                model,
                adata[adata.X.sum(1) > 500],
                how="random expr",
                preprocess="softmax",
                head_agg="max",
                filtration="none",
                forward_mode="none",
                num_workers=8,
                num_genes=2200,
                max_cells=maxcells,
                doplot=False,
                batch_size=batch_size,
                devices=1,
            )

            grn = grn_inferer(layer=layers, cell_type=celltype)
            grn.var.index = make_index_unique(grn.var["symbol"].astype(str))
            metrics[celltype + "_scprint"] = BenGRN(
                grn, doplot=False
            ).scprint_benchmark()
            del grn
            gc.collect()
            grn_inferer = GRNfer(
                model,
                adata[adata.X.sum(1) > 500],
                how="most var across",
                preprocess="softmax",
                head_agg="none",
                filtration="none",
                forward_mode="none",
                num_workers=8,
                num_genes=maxgenes,
                max_cells=maxcells,
                doplot=False,
                batch_size=batch_size,
                devices=1,
            )
            grn = grn_inferer(layer=layers, cell_type=celltype)
            grn.var.index = make_index_unique(grn.var["symbol"].astype(str))
            grn.varp["all"] = grn.varp["GRN"]
            grn.varp["GRN"] = grn.varp["GRN"].mean(-1)
            metrics[celltype + "_scprint_mean"] = BenGRN(
                grn, doplot=False
            ).scprint_benchmark()
            if clf_omni is None:
                grn.varp["GRN"] = grn.varp["all"]
                _, m, clf_omni = train_classifier(
                    grn,
                    C=1,
                    train_size=0.6,
                    max_iter=300,
                    class_weight={1: 800, 0: 1},
                    return_full=False,
                    shuffle=True,
                    doplot=False,
                )
                joblib.dump(clf_omni, "clf_omni.pkl")
                metrics["classifier"] = m
            grn.varp["GRN"] = grn.varp["all"][:, :, clf_omni.coef_[0] > 0].mean(-1)
            metrics[celltype + "_scprint_class"] = BenGRN(
                grn, doplot=False
            ).scprint_benchmark()
            del grn
            gc.collect()
    return metrics
