
# scprint: Large Cell Model for scRNAseq data

[![PyPI version](https://badge.fury.io/py/scprint.svg)](https://badge.fury.io/py/scprint)
[![Documentation Status](https://readthedocs.org/projects/scprint/badge/?version=latest)](https://scprint.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/scprint)](https://pepy.tech/project/scprint)
[![Downloads](https://pepy.tech/badge/scprint/month)](https://pepy.tech/project/scprint)
[![Downloads](https://pepy.tech/badge/scprint/week)](https://pepy.tech/project/scprint)
[![GitHub issues](https://img.shields.io/github/issues/jkobject/scPRINT)](https://img.shields.io/github/issues/jkobject/scPRINT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/391909874.svg)]()

![logo](logo.png)

scPRINT is a large transformer model built for the inference of gene network (connections between genes explaining the cell's expression profile) from scRNAseq data.

It uses novel encoding and decoding of the cell expression profile as well as new pre-training methodologies to learn a cell model.

scPRINT can do lots of things:

- __expression denoising__: increase the resolution of your scRNAseq data
- __cell embedding__: generate a low-dimensional representation of your dataset
- __label prediction__: predict the cell type, disease, sequencer, sex, and ethnicity of your cells
- __gene network inference__: generate a gene network from any cell or cell cluster in your scRNAseq dataset

[Read the paper!]() if you want to know more about scPRINT.

![figure1](figure1.png)

## Install it from PyPI

If you want to be using flashattention2, know that it only supports triton 2.0 MLIR's version and torch==2.0.0 for now.

👷 WIP ...

<!---

```bash
pip install 'lamindb[jupyter,bionty]'
```

then install scPrint

```bash
pip install scprint
```
> if you have a GPU that you want to use, you will benefit from flashattention. and you will have to do some more specific installs:

1. find the version of torch 2.0.0 / torchvision 0.15.0 / torchaudio 2.0.0 that match your nvidia drivers on the torch website.
2. apply the install command
3. do `pip install pytorch-fast-transformers torchtext==0.15.1`
4. do `pip install triton==2.0.0.dev20221202 --no-deps`

You should be good to go. You need those specific versions for everything to work...

This is not my fault, scream at nvidia :wink:
-->

## Install it in dev mode

For the moment scPRINT has been tested on MacOS and Linux (Ubuntu 20.04) with Python 3.10.

If you want to be using flashattention2, know that it only supports triton 2.0 MLIR's version and torch==2.0.0 for now.


```python
conda create -n "[whatever]" python==3.10
git clone https://github.com/jkcobject/scPRINT
git clone https://github.com/jkobject/GRnnData
git clone https://github.com/jkobject/benGRN
cd scPRINT
git checkout dev
git submodule init
git submodule update
pip install 'lamindb[jupyter,bionty]'
pip install -e scDataloader
pip install -e ../GRnnData/
pip install -e ../benGRN/
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
# install the dev tooling if you need it too
pip install -e ".[dev]"
pip install -r requirements-dev.txt
pip install triton==2.0.0.dev20221202 --no-deps # only if you have a compatible gpu (e.g. not available for apple GPUs for now, see https://github.com/triton-lang/triton?tab=readme-ov-file#compatibility)
# install triton as mentioned in .toml if you want to
mkdocs serve # to view the dev documentation
```

We use additional packages we developped, refer to their documentation for more information:

- [scDataLoader](https://github.com/jkobject/scDataLoader): a dataloader for training large cell models.
- [GRnnData](https://github.com/cantinilab/GRnnData): a package to work with gene networks from single cell data.
- [benGRN](https://github.com/jkobject/benGRN): a package to benchmark gene network inference methods from single cell data.

### lamin.ai

⚠️ if you want to use the scDataloader's multi dataset mode or if you want to preprocess datasets and other functions of the model, you will need to use lamin.ai.

In that case connect with google or github to [lamin.ai](https://lamin.ai/login), then be sure to connect before running anything (or before starting a notebook): `lamin login <email> --key <API-key>`. Follow the instructions on [their website](https://docs.lamin.ai/guide).

## Usage

### scPRINT's basic commands

This is the most minimal example of how scprint gets used:

```py
from lightning.pytorch import Trainer
from scprint import scPrint
from scdataloader import DataModule

datamodule = DataModule(...)
model = scPrint(...)
trainer = Trainer(...)
trainer.fit(model, datamodule=datamodule)
...
```

or

```bash
$ scprint fit/train/predict/test --config config/[medium|large|vlarge] ...
```

### Notes on GPU/CPU usage with triton

If you do not have [triton](https://triton-lang.org/main/python-api/triton.html) installed you will not be able to take advantage of gpu acceleration, but you can still use the model on the cpu.

In that case, if loading from a checkpoint that was trained with flashattention, you will need to specify `transformer="normal"` in the `load_from_checkpoint` function like so:

```python
model = scPrint.load_from_checkpoint(
    '../data/temp/last.ckpt', precpt_gene_emb=None,
    transformer="normal")
```

We now explore the different usages of scPRINT:

### I want to generate gene networks from scRNAseq data:

-> refer to the section 1. gene network inference in [this notebook](./notebooks/cancer_usecase.ipynb#).

-> more examples in this notebook [./notebooks/assessments/bench_omni.ipynb](./notebooks/assessments/bench_omni.ipynb).

### I want to generate cell embeddings and cell label predictions from scRNAseq data:

-> refer to the embeddings and cell annotations section in [this notebook](./notebooks/cancer_usecase.ipynb).

### I want to denoising my scRNAseq dataset:

-> refer to the Denoising of B-cell section in [this notebook](./notebooks/cancer_usecase.ipynb).

-> More example in our benchmark notebook [./notebooks/assessments/bench_denoising.ipynb](./notebooks/assessments/bench_denoising.ipynb).

### I want to generate an atlas level embedding

-> refer to the notebook [nice_umap.ipynb](./figures/nice_umap.ipynb).

### Documentation

/!\ WIP /!\

<!-- 
for more information on usage please see the documentation in [https://www.jkobject.com/scPrint/](https://www.jkobject.com/scPrint/)

-->

### Model Weights

Model weights are available on [hugging face](https://huggingface.co/jkobject).

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

Read the [training runs](https://wandb.ai/ml4ig/scprint_scale/reports/scPRINT-trainings--Vmlldzo4ODIxMjgx?accessToken=80metwx7b08hhourotpskdyaxiflq700xzmzymr6scvkp69agybt79l341tv68hp) document to know more about how training was performed and the results there.

acknowledgement:
[python template](https://github.com/rochacbruno/python-project-template)
[laminDB](https://lamin.ai/)
[lightning](https://lightning.ai/)

Awesome Large Cell Model created by Jeremie Kalfon.
