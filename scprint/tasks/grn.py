import scanpy as sc
import torch

from scdataloader.data import SimpleAnnDataset
from scdataloader import Collator
from grnndata import utils
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer

from typing import List
from anndata import AnnData

from scprint.utils.sinkhorn import SinkhornDistance
from scprint.utils import load_genes
from scprint.utils import load_genes

from grnndata import GRNAnnData


class GRNfer:
    def __init__(
        self,
        model: torch.nn.Module,
        adata: AnnData,
        batch_size: int = 64,
        num_workers: int = 8,
        num_genes: int = 3000,
        precision: str = "16-mixed",
        organisms: List[str] = [
            "NCBITaxon:9606",
        ],
        cell_type_col="cell_type",
        model_name: str = "scprint",
        preprocess="sinkhorn",
        cell_type_col="cell_type",
        genes: list = [],
        loc="./",
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
        if how == "random expr" and cell_agg == "consensus":
            raise ValueError("cannot have random expr with consensus")
        self.how = how
        assert self.how in [
            "most var within",
            "most var across",
            "random expr",
            "given",
        ], "how must be one of 'most var within', 'most var across', 'random expr', 'given'"
        self.num_genes = num_genes
        self.organisms = organisms
        self.model_name = model_name
        self.adata = adata
        self.preprocess = preprocess
        self.cell_type_col = cell_type_col
        self.filtration = filtration
        self.cell_agg = cell_agg
        self.doplot = doplot
        self.genes = genes
        self.trainer = Trainer(precision=precision)
        # subset_hvg=1000, use_layer='counts', is_symbol=True,force_preprocess=True, skip_validate=True)

    def __call__(self, layer, cell_type=None):
        # Add at least the organism you are working with
        if cell_type is not None:
            subadata = self.adata[
                self.adata.obs[self.cell_type_col] == cell_type
            ].copy()[:1]
        else:
            subadata = (
                self.adata.copy()[: self.max_cells]
                if self.max_cells
                else self.adata.copy()
            )
        if self.how == "most var within":
            sc.pp.highly_variable_genes(
                subadata, n_top_genes=self.num_genes, flavor="seurat_v3"
            )
            self.curr_genes = subadata.var.index[subadata.var.highly_variable].tolist()
            print(
                "number of expressed genes in this cell type: "
                + str((subadata.X.sum(0) > 1).sum())
            )
        elif self.how == "most var across" and cell_type is not None:
            sc.tl.rank_genes_groups(
                self.adata, groupby=self.cell_type_col, groups=[cell_type]
            )
            self.curr_genes = self.adata.uns["rank_genes_groups"]["names"][cell_type][
                : self.num_genes
            ].tolist()
        elif self.how == "random expr":
            self.curr_genes = self.model.genes
            # raise ValueError("cannot do it yet")
            pass
        elif self.how == "given" and len(self.genes) > 0:
            self.curr_genes = self.genes
        else:
            raise ValueError("how must be one of 'most var', 'random expr'")

        adataset = SimpleAnnDataset(
            subadata, obs_to_output=["organism_ontology_term_id"]
        )
        col = Collator(
            organisms=self.organisms,
            valid_genes=self.model.genes,
            how="some" if self.how != "random expr" else "random expr",
            genelist=self.curr_genes if self.how != "random expr" else [],
        )
        dataloader = DataLoader(
            adataset,
            collate_fn=col,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        self.model.get_attention_layer = layers
        self.trainer.predict(self.model, dataloader)

        attn = self.model.mean_attn[0][0:][:, 0, :, :].permute(
            1, 0, 2
        ) @ self.model.mean_attn[0][0:][:, 1, :, :].permute(1, 2, 0)
        return attn
        scale = self.model.mean_attn[0].shape[-1] ** -0.5
        attn = attn * scale
        if self.preprocess == "sinkhorn":
            if attn.numel() > 100_000_000:
                raise ValueError("you can't sinkhorn such a large matrix")
            sink = SinkhornDistance(0.1, max_iter=200)
            attn = sink(attn)[0]
            attn = attn * attn.shape[-1]
        elif self.preprocess == "softmax":
            attn = torch.nn.functional.softmax(attn, dim=-1)
        elif self.preprocess == "none":
            pass
        else:
            raise ValueError("preprocess must be one of 'sinkhorn', 'softmax', 'none'")

        if agg == "mean":
            a = a.mean(0).detach().cpu().numpy()
        elif agg == "max":
            a = a.max(0)[0].detach().cpu().numpy()
        elif agg == "None":
            a = a.detach().cpu().numpy()
            grns = []
            for suba in a:
                print(
                    "avg link count: "
                    + str((suba > (1 / attn.shape[-1])).sum() / attn.shape[-1])
                )
                suba[suba < (1 / a.shape[-1])] = 0
                grn = GRNAnnData(
                    subadata[:, col.accepted_genes[self.organisms[0]]][
                        :, col.to_subset[self.organisms[0]]
                    ].copy(),
                    grn=suba,
                )
                top_central_genes = utils.get_centrality(grn)
                grn.var.loc[[i[0] for i in top_central_genes], "symbol"]

                grn.var_names = grn.var["symbol"]
                grn.var["TFs"] = [
                    True if i in utils.TF else False for i in grn.var_names
                ]

                grn.var = grn.var.drop(
                    columns=["stable_id", "created_at", "updated_at"]
                )
        else:
            raise ValueError("agg must be one of 'mean', 'max' or 'None'")
        print(
            "avg link count: " + str((a > (1 / attn.shape[-1])).sum() / attn.shape[-1])
        )
        if self.filtration == "thresh":
            a[a < (1 / a.shape[-1])] = 0
        elif self.filtration == "none":
            pass
        elif self.filtration == "top-k":
            np.argsort(a)
        elif self.filtration == "known":
            pass
        else:
            raise ValueError("filtration must be one of 'thresh', 'none' or 'top-k'")
        grn = GRNAnnData(
            subadata[:, subadata.var.index.isin(self.curr_genes)].copy(),
            grn=grn,
        )
        grn = grn[:, (grn.X != 0).sum(0) > (self.max_cells / 20)]

        grn.var_names = grn.var["symbol"]
        grn.var["TFs"] = [True if i in utils.TF else False for i in grn.var_names]
        grn.uns["grn_scprint_params"] = {
            "filtration": self.filtration,
            "how": self.how,
            "cell_agg": self.cell_agg,
            "preprocess": self.preprocess,
            "head_agg": self.head_agg,
        }
        grn.write_h5ad(loc + "grn_fromscprint.h5ad")
        return from_anndata(grn)


def get_GTdb(db="omnipath"):
    if db == "omnipath":
        if not os.path.exists(FILEDIR + "/../../data/main/omnipath.parquet"):
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
            da = np.zeros((len(varnames), len(varnames)), dtype=np.float)
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
