import numpy as np
import pandas as pd
import lnschema_bionty as lb
from scdataloader.utils import load_genes
from torch import Tensor

class Collator:
    def __init__(
        self,
        organisms: list,
        org_to_id: dict,
        labels: list = [],
        genelist: list = [],
        max_len=2000,
        n_bins=0,
        add_zero_genes=200,
        logp1=True,
        norm_to=1e4,
        how="most expr"
    ):
        self.organisms = organisms
        self.genelist = genelist
        self.max_len = max_len
        self.n_bins = n_bins
        self.add_zero_genes = add_zero_genes
        self.logp1 = logp1
        self.norm_to = norm_to
        self.org_to_id = org_to_id
        self.how = how
        self.organism_ids = set(self.org_to_id.values())

        self.start_idx = {}
        self.accepted_genes = {}
        self.genedf = load_genes(self.organisms)
        for organism in set(self.genedf.organism):
            ogenedf = self.genedf[self.genedf.organism == organism]
            self.start_idx.update(
                {
                    self.org_to_id[organism]: np.where(
                        self.genedf.organism == organism
                    )[0][0]
                }
            )
            if len(self.genelist) > 0:
                self.accepted_genes.update(
                    {self.org_to_id[organism]: ogenedf.index.isin(self.genelist)}
                )
        self.labels = labels
        self.tt_counts_idx = self.labels.index("total_counts")
        self.tp_idx = self.labels.index("heat_diff")
        self.org_idx = self.labels.index("organism_ontology_term_id")

    def __call__(self, batch):
        # do count selection
        # get the unseen info and don't add any unseen
        # get the I most expressed genes, add randomly some unexpressed genes that are not unseen
        exprs = []
        total_count = []
        other_classes = []
        gene_locs = []
        tp = []
        for elem in batch:
            organism_id = elem[6]
            if organism_id not in self.organism_ids:
                continue
            expr = np.array(elem[0])
            expr = expr[self.accepted_genes[organism_id]]
            if self.how == "most expr":
                loc = np.argsort(expr)[-(self.max_len) :][::-1]
            if self.how == "random expr":
                nnz_loc = np.where(expr > 0)[0]
                loc = nnz_loc[np.random.choice(len(nnz_loc), self.max_len, replace=False)]
            if self.add_zero_genes > 0:
                zero_loc = np.where(expr == 0)[0]
                zero_loc = [
                    np.random.choice(len(zero_loc), self.add_zero_genes, replace=False)
                ]
                loc = np.concatenate((loc, zero_loc), axis=None)
            exprs.append(expr[loc])
            gene_locs.append(loc + self.start_idx[organism_id])

            total_count.append(elem[-3])
            tp.append(elem[-4])

            other_classes.append(elem[1:-4])

        expr = np.array(exprs)
        tp = np.array(tp)
        gene_locs = np.array(gene_locs)
        total_count = np.array(total_count)
        other_classes = np.array(other_classes)

        # normalize counts
        if self.norm_to is not None:
            expr = (expr * self.norm_to) / total_count[:, None]
        if self.logp1:
            expr = np.log2(1 + expr)

        # do binning of counts
        if self.n_bins:
            pass

        # find the associated gene ids (given the species)

        # get the NN cells

        # do encoding / selection a la scGPT

        # do encoding of graph location
        # encode all the edges in some sparse way
        # normalizing total counts between 0,1
        return [Tensor(expr), Tensor(gene_locs).int(), Tensor(other_classes).int(), Tensor(tp), Tensor(total_count)]


class GeneformerCollator(Collator):
    def __init__(self, *args, gene_norm_list: list, **kwargs):
        super().__init__(*args, **kwargs)
        self.gene_norm_list = gene_norm_list

    def __call__(self, batch):
        super().__call__(batch)
        # normlization per gene

        # tokenize the empty locations


class scGPTCollator(Collator):
    def __call__(self, batch):
        super().__call__(batch)
        # binning

        # tokenize the empty locations


class scPRINTCollator(Collator):
    def __call__(self, batch):
        super().__call__(batch)
