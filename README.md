
# scprint: Large Cell Model for scRNAseq data

[![PyPI version](https://badge.fury.io/py/scprint.svg)](https://badge.fury.io/py/scprint)
[![Documentation Status](https://readthedocs.org/projects/scprint/badge/?version=latest)](https://scprint.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/scprint)](https://pepy.tech/project/scprint)
[![Downloads](https://pepy.tech/badge/scprint/month)](https://pepy.tech/project/scprint)
[![Downloads](https://pepy.tech/badge/scprint/week)](https://pepy.tech/project/scprint)
[![GitHub issues](https://img.shields.io/github/issues/jkobject/scPRINT)](https://img.shields.io/github/issues/jkobject/scPRINT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/391909874.svg)](https://zenodo.org/badge/latestdoi/391909874)

![logo](logo.png)

scPRINT is a novel transformer model for the inference of gene regulatory network from scRNAseq data. It uses novel encoding and decoding schemes as well as new pre-training methodologies to learn a model of the cell. But scPRINT can do lots of things: [Read the paper!]()

![figure1](figure1.png)

## Install it from PyPI

If you want to be using flashattention2, know that it only supports torch==2.0.0 for now.

🚨 **Important Notice:** Only the **development install** currently works (see [dev mode](#in-dev-mode))! 🚨

```bash
pip install 'lamindb[jupyter,bionty]'
```

then install scPrint

```bash
pip install scprint
```
> if you have a GPU that you want to use, you will benefit from flashattention. and you will have to do some more specific installs:

1. find the version of torch 2.0.0 / torchvision 0.15.0 / torchaudio 2.0.0 that match your nvidia drivers in the torch website.
2. apply the install command
3. do `pip install pytorch-fast-transformers torchtext==0.15.1`
4. do `pip install triton==2.0.0.dev20221202 --no-deps`

You should be good to go. You need those specific versions for everything to work.. 
not my fault, scream at nvidia, pytorch, Tri Dao and OpenAI :wink:


### in dev mode

```python
conda create ...
git clone https://github.com/jkobject/scPRINT
git clone https://github.com/jkobject/GRnnData
git clone https://github.com/jkobject/benGRN
cd scPRINT
git checkout dev
git submodule init
git submodule update
pip install -e scDataloader
pip install -e ../GRnnData/
pip install -e ../benGRN/
tall torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
# install pytorch as mentionned above if you have a GPU
pip install -e .[dev]
pip install 'lamindb[jupyter,bionty]'
pip install triton==2.0.0.dev20221202 --no-deps
```

## Usage

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

for more information on usage please see the documentation in [https://www.jkobject.com/scPrint/](https://www.jkobject.com/scPrint/)

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

acknowledgement:
[python template](https://github.com/rochacbruno/python-project-template)
[laminDB](https://lamin.ai/)
[Lightning](https://lightning.ai/)

Awesome Large Cell Model created by Jeremie Kalfon.

