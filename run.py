import lamindb as ln
import lnschema_bionty as lb

import pandas as pd

from lightning.pytorch import Trainer, seed_everything

from scprint import scPrint
from scprint.utils import getBiomartTable

from scdataloader import Dataset
from scdataloader import DataModule
from scprint.dataloader import embed
from scdataloader.utils import load_genes
from scprint.dataloader import Collator

import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler

torch.set_float32_matmul_precision("medium")
seed_everything(42, workers=True)
lb.settings.organism = "human"

embeddings = pd.read_parquet("./data/temp/embeddings.parquet")
embeddings.columns = ["emb_" + str(i) for i in embeddings.columns]
# and annotations
biomart = getBiomartTable(attributes=["start_position", "chromosome_name"]).set_index(
    "ensembl_gene_id"
)
biomart = biomart.loc[~biomart.index.duplicated(keep="first")]
biomart = biomart.sort_values(by=["chromosome_name", "start_position"])
# and location
c = []
i = 0
prev_position = -100000
prev_chromosome = None
for _, r in biomart.iterrows():
    if (
        r["chromosome_name"] != prev_chromosome
        or r["start_position"] - prev_position > 10_000
    ):
        i += 1
    c.append(i)
    prev_position = r["start_position"]
    prev_chromosome = r["chromosome_name"]
print(f"reduced the size to {len(set(c))/len(biomart)}")
biomart["pos"] = c

name = "preprocessed dataset"
dataset = ln.Collection.filter(name=name).first()
dataset.artifacts.count()

# TODO: drop tissue & dev stage until part or is taken in account

hierarchical_labels = [
    "cell_type_ontology_term_id",
    # "tissue_ontology_term_id",
    "disease_ontology_term_id",
    # "development_stage_ontology_term_id",
    "assay_ontology_term_id",
    "self_reported_ethnicity_ontology_term_id",
]

labels_weighted_sampling = hierarchical_labels + [
    "sex_ontology_term_id",
    "organism_ontology_term_id",
]

all_labels = labels_weighted_sampling + [
    #'dataset_id',
    #'cell_culture',
    "heat_diff",
    "total_counts",
    "nnz",
    "dpt_group",
]
mdataset = Dataset(
    dataset,
    organisms=["NCBITaxon:9606"],
    obs=all_labels,
    clss_to_pred=labels_weighted_sampling,
    hierarchical_clss=hierarchical_labels,
)
print(mdataset)

d_model = 128
m = torch.nn.AdaptiveAvgPool1d(d_model)
sembeddings = pd.DataFrame(
    data=m(torch.tensor(embeddings.values)),
    index=embeddings.index,
    columns=[f"emb_{i}" for i in range(d_model)],
)
labels = {k: len(v) for k, v in mdataset.class_topred.items()}

cls_hierarchies = {}
for k, dic in mdataset.class_groupings.items():
    rdic = {}
    for sk, v in dic.items():
        rdic[mdataset.encoder[k][sk]] = [mdataset.encoder[k][i] for i in list(v)]
    cls_hierarchies[k] = rdic

df = sembeddings.join(biomart, how="inner")

genedf = load_genes(["NCBITaxon:9606"])
df = df.loc[genedf[genedf.index.isin(df.index)].index]

# we might want not to order the genes by expression (or do it?)
# we might want to not introduce zeros and
col = Collator(
    organisms=[
        "NCBITaxon:9606",
    ],
    valid_genes=df.index.tolist(),
    max_len=1000,
    add_zero_genes=100,
    org_to_id={
        "NCBITaxon:9606": mdataset.encoder["organism_ontology_term_id"][
            "NCBITaxon:9606"
        ]
    },
    tp_name="heat_diff",
    organism_name="organism_ontology_term_id",
    class_names=labels_weighted_sampling,
)  # mdataset.encoder['organism_ontology_term_id'])

datamodule = DataModule(
    mdataset,
    label_to_weight=labels_weighted_sampling,
    collate_fn=col,
    batch_size=64,
    num_workers=8,
)
datamodule.setup()
for i in datamodule.train_dataloader():
    break

model = scPrint(
    genes=df.index.tolist(),
    d_model=d_model,
    nhead=4,
    d_hid=d_model,
    nlayers=4,
    layers_cls=[],
    labels=labels,
    cls_hierarchy=cls_hierarchies,
    dropout=0.1,
    transformer="flash",
    use_precpt_gene_emb=df.iloc[:, :d_model].values.astype(float),
    gene_pos_enc=df["pos"].tolist(),
    mvc_decoder="inner product",
)
# create a function to transform an scGPT checkpoint to an scPrint's
# ckpt = torch.load("../../scGPT/save/model_e6.pt")
# scPrint.load_from_checkpoint("../../scGPT/save/model_e6.pt")

wandb_logger = WandbLogger(project="scprint_test", save_dir="./data/tensorboard")
wandb_logger.watch(model)

# tlogger = TensorBoardLogger(save_dir="../data/tensorboard")

# pytorch_prof = PyTorchProfiler(
#    "./data/tensorboard",
#    emit_nvtx=False,
#    group_by_input_shape=True,
#    record_shapes=True,
#    profile_memory=True,
#    with_stack=True,
#    on_trace_ready=torch.profiler.tensorboard_trace_handler("./data/tensorboard/"),
# )
# sets seeds for numpy, torch and python.random.
from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=-1)

trainer = Trainer(
    precision=16,
    profiler="simple",
    callbacks=[checkpoint_callback],
    gradient_clip_val=10,
    max_time={"hours": 4},
    limit_train_batches=20000,
    limit_test_batches=0.3,
    limit_val_batches=4000,
    logger=wandb_logger,
)

# model.labels = {}
# model.expr_decoder.nfirst_labels_to_skip=3
trainer.fit(model, datamodule=datamodule)
