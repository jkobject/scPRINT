from anndata import AnnData


class GRNAnnData(AnnData):
    def __init__(self, grn, **kwargs):
        super().__init__(**kwargs)

        self.obsp["GRN"] = grn


def from_anndata(adata):
    if "GRN" not in adata.obsp:
        raise ValueError("GRN not found in adata.obsp")
    return GRNAnnData(adata.obsp["GRN"], X=adata)


def get_centrality(GRNAnnData, k=30):
    """
    get_centrality uses the networkx library to calculate the centrality of each node in the GRN.
    The centrality is added to the GRNAnnData object as a new column in the var dataframe.
    also prints the top K most central nodes in the GRN.

    Args:
        GRNAnnData (_type_): _description_
    """
    import networkx as nx

    G = nx.from_scipy_sparse_matrix(GRNAnnData.obsp["GRN"])
    centrality = nx.eigenvector_centrality(G)

    GRNAnnData.var["centrality"] = [
        centrality.get(gene, 0) for gene in GRNAnnData.var_names
    ]

    top_central_genes = sorted(
        [(node, centrality) for node, centrality in centrality.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:k]
    print("Top central genes:", top_central_genes)


def enrichment(GRNAnnData, of="Targets", for_="TFs", doplot=True, **kwargs):
    """
    enrichment uses the gseapy library to calculate the enrichment of the target genes in the adata
    the enrichment is returned and plotted

    Args:
        GRNAnnData (_type_): _description_
        of (str, optional): _description_. Defaults to "Targets".
        for_ (str, optional): _description_. Defaults to "TFs".
    """
    import gseapy as gp
    from gseapy.plot import barplot, dotplot

    mapping = {
        "TFs": "KEGG_2019_Human",
    }

    # define gene sets
    if of == "Targets":
        gene_sets = GRNAnnData.var_names
    elif of == "TFs":
        gene_sets = GRNAnnData.var["TFs"]
    else:
        raise ValueError("of must be one of 'Targets', 'TFs'")

    # run enrichment analysis
    enr = gp.enrichr(
        gene_list=gene_sets, description=for_, gene_sets=mapping[for_], **kwargs
    )

    # plot results
    if doplot:
        barplot(enr.res2d, title=for_)

    return enr


def similarity(GRNAnnData, other_GRNAnnData):
    pass


def get_subnetwork(GRNAnnData, on="TFs"):
    if type(on) is list:
        pass
    elif on == "TFs":
        pass
    elif on == "Regulators":
        pass
    else:
        raise ValueError("on must be one of 'TFs', 'Regulators', or a list of genes")
    pass


def focuses_more_on(GRNAnnData, on="TFs"):
    pass
