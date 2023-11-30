from scprint import utils
from scprint.data_loader import PROTBERT, RNABERT
import subprocess
from torch.nn.functional import avg_pool1d
import os


def embed(
    genesdf,
    organism="homo_sapiens",
    cache=True,
    fasta_path="/tmp/data/fasta/",
    subfasta_path="/tmp/subset.fa",
    config="esm-extract",
    pretrained_model="esm2_t33_650M_UR50D",
):
    # given a gene file and organism
    # load the organism fasta if not already done
    utils.load_fasta_species(species=organism, output_path=fasta_path, cache=cache)
    # subset the fasta
    fasta_file = next(
        file for file in os.listdir(fasta_path) if file.endswith(".all.fa.gz")
    )
    protgenedf = genesdf[genesdf["biotype"] == "protein_coding"]
    subprocess.run(["gunzip", fasta_file], check=True)
    utils.subset_fasta(
        protgenedf["ensembl_gene_id"].tolist(),
        subfasta_path=fasta_path + "subset.fa",
        fasta_path=fasta_file,
        drop_unknown_seq=True,
    )
    # subset the gene file
    # embed
    prot_embeder = PROTBERT(config=config, pretrained_model=pretrained_model)
    prot_embeddings = prot_embeder(
        fasta_path + "subset.fa", output_folder=fasta_path + "esm_out/", cache=cache
    )
    # load the data and erase / zip the rest
    subprocess.run(["gzip", fasta_file], check=True)
    # return the embedding and gene file
    # do the same for RNA
    rnagenedf = genesdf[genesdf["biotype"] != "protein_coding"]
    fasta_file = next(
        file for file in os.listdir(fasta_path) if file.endswith(".ncrna.fa.gz")
    )
    utils.subset_fasta(
        rnagenedf["ensembl_gene_id"].tolist(),
        subfasta_path=subfasta_path,
        fasta_path=fasta_file,
        drop_unknown_seq=True,
    )
    rna_embeder = RNABERT(config=config, pretrained_model=pretrained_model)
    rna_embeddings = rna_embeder()
    # Check if the sizes of the embeddings are not the same
    if len(prot_embeddings) != len(rna_embeddings):
        # If not, make them the same size using average pooling of the larger one
        if len(prot_embeddings) > len(rna_embeddings):
            prot_embeddings = avg_pool1d(
                prot_embeddings, len(prot_embeddings) - len(rna_embeddings)
            )
        else:
            rna_embeddings = avg_pool1d(
                rna_embeddings, len(rna_embeddings) - len(prot_embeddings)
            )
    # Concatenate the embeddings
    return torch.cat([prot_embeddings, rna_embeddings], dim=1)
