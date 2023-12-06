from scprint import utils
from scprint.utils.utils import run_command
from scprint.dataloader import PROTBERT, RNABERT
from torch.nn import AdaptiveAvgPool1d
import pandas as pd
import torch
import os


def embed(
    genedf,
    organism="homo_sapiens",
    cache=True,
    fasta_path="/tmp/data/fasta/",
    embedding_size=512,
):
    # given a gene file and organism
    # load the organism fasta if not already done
    utils.load_fasta_species(species=organism, output_path=fasta_path, cache=cache)
    # subset the fasta
    fasta_file = next(
        file for file in os.listdir(fasta_path) if file.endswith(".all.fa.gz")
    )
    protgenedf = genedf[genedf["biotype"] == "protein_coding"]
    utils.utils.run_command(["gunzip", fasta_path + fasta_file])
    utils.subset_fasta(
        protgenedf["ensembl_gene_id"].tolist(),
        subfasta_path=fasta_path + "subset.fa",
        fasta_path=fasta_path + fasta_file[:-3],
        drop_unknown_seq=True,
    )
    # subset the gene file
    # embed
    prot_embedder = PROTBERT()
    import pdb

    pdb.set_trace()
    prot_embeddings = prot_embedder(
        fasta_path + "subset.fa", output_folder=fasta_path + "esm_out/", cache=cache
    )
    # load the data and erase / zip the rest
    utils.utils.run_command(["gzip", fasta_path + fasta_file[:-3]])
    # return the embedding and gene file
    # do the same for RNA
    rnagenedf = genedf[genedf["biotype"] != "protein_coding"]
    fasta_file = next(
        file for file in os.listdir(fasta_path) if file.endswith(".ncrna.fa.gz")
    )
    utils.utils.run_command(["gunzip", fasta_path + fasta_file])
    utils.subset_fasta(
        rnagenedf["ensembl_gene_id"].tolist(),
        subfasta_path=fasta_path + "subset.ncrna.fa",
        fasta_path=fasta_path + fasta_file[:-3],
        drop_unknown_seq=True,
    )
    rna_embedder = RNABERT()
    rna_embeddings = rna_embedder(fasta_path + "subset.ncrna.fa")
    # Check if the sizes of the cembeddings are not the same
    utils.utils.run_command(["gzip", fasta_path + fasta_file[:-3]])

    m = AdaptiveAvgPool1d(embedding_size)
    prot_embeddings = pd.DataFrame(
        data=m(torch.tensor(prot_embeddings.values)), index=prot_embeddings.index
    )
    rna_embeddings = pd.DataFrame(
        data=m(torch.tensor(rna_embeddings.values)), index=rna_embeddings.index
    )
    # Concatenate the embeddings
    return pd.concat([prot_embeddings, rna_embeddings])
