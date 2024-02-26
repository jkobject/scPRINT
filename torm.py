import scanpy as sc
import numpy as np
from moscot.problems.time import TemporalProblem
from cellrank.kernels import PseudotimeKernel

import os


def reprocess(adata):
    adata.layers["clean"] = sc.pp.log1p(
        sc.pp.normalize_total(adata, target_sum=1e4, inplace=False)["X"]
    )
    adata.obsm["clean_pca"] = sc.pp.pca(
        adata.layers["clean"],
        n_comps=300 if adata.shape[0] > 300 else adata.shape[0] - 2,
    )
    sc.pp.neighbors(adata, use_rep="clean_pca")
    sc.tl.leiden(adata, key_added="leiden_3", resolution=3.0)
    sc.tl.leiden(adata, key_added="leiden_2", resolution=2.0)
    sc.tl.leiden(adata, key_added="leiden_1", resolution=1.0)
    sc.tl.umap(adata)
    sc.tl.diffmap(adata)
    # create a meta group
    adata.obs["dpt_group"] = (
        adata.obs["disease_ontology_term_id"].astype(str)
        + "_"
        + adata.obs["cell_type_ontology_term_id"].astype(str)
        + "_"
        + adata.obs["tissue_ontology_term_id"].astype(str)
    )  # + "_" + adata.obs['dataset_id'].astype(str)

    # if group is too small
    okgroup = [i for i, j in adata.obs["dpt_group"].value_counts().items() if j >= 10]
    not_okgroup = [i for i, j in adata.obs["dpt_group"].value_counts().items() if j < 3]
    # set the group to empty
    adata.obs.loc[adata.obs["dpt_group"].isin(not_okgroup), "dpt_group"] = ""
    adata.obs["heat_diff"] = np.nan

    if adata.obs["organism_ontology_term_id"].iloc[0] in [
        "NCBITaxon:9606",
        "NCBITaxon:10090",
    ]:
        org = (
            "human"
            if adata.obs["organism_ontology_term_id"].iloc[0] == "NCBITaxon:9606"
            else "mouse"
        )
        adata.var["ensembl_gene_id"] = adata.var.index
        adata.var = adata.var.set_index("symbol")
        adata.var.index = adata.var.index.astype(str)
        adata.var_names_make_unique()
        TemporalProblem(adata).score_genes_for_marginals(
            gene_set_proliferation=org, gene_set_apoptosis=org
        )
        prior_growth = np.exp(
            (adata.obs["proliferation"].values - adata.obs["apoptosis"].values) / 2
        )
        adata.obs["normalized_growth"] = prior_growth
        adata.var["symbol"] = adata.var.index
        adata.var = adata.var.set_index("ensembl_gene_id")
        # for each group
        for val in set(okgroup):
            if val == "":
                continue
            # get the best root cell
            eq = adata.obs.dpt_group == val
            loc = np.where(eq)[0]

            root_ixs = loc[adata.obs["normalized_growth"][eq].argmax()]
            adata.uns["iroot"] = root_ixs
            # compute the diffusion pseudo time from it
            sc.tl.dpt(adata)
            adata.obs.loc[eq, "heat_diff"] = adata.obs.loc[eq, "dpt_pseudotime"]
            adata.obs.drop(columns=["dpt_pseudotime"], inplace=True)

        # sort so that the next time points are aligned for all groups
        PseudotimeKernel(
            adata, time_key="normalized_growth"
        ).compute_transition_matrix().write_to_adata()
        adata = adata[adata.obs.sort_values(["dpt_group", "heat_diff"]).index]
    else:
        adata.obs["dpt_group"] = ""
    return adata


files = os.listdir("/home/ml4ig1/scprint/.lamindb/")

for i, f in enumerate(files[51:]):
    print(" ")
    print(" ")
    adata = reprocess(sc.read_h5ad("/home/ml4ig1/scprint/.lamindb/" + f))
    print(i)
    print(adata)
    sc.write("/home/ml4ig1/scprint/.lamindb/" + f, adata)
    del adata
