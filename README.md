
# scprint

[![codecov](https://codecov.io/gh/jkobject/scPRINT/branch/main/graph/badge.svg?token=scPRINT_token_here)](https://codecov.io/gh/jkobject/scPRINT)
[![CI](https://github.com/jkobject/scPRINT/actions/workflows/main.yml/badge.svg)](https://github.com/jkobject/scPRINT/actions/workflows/main.yml)

Awesome Large Transcriptional Model created by Jeremie Kalfon

scprint = single cell pretrained regulation inference neural network from transcripts

using: 


## Install it from PyPI

first have a good version of pytorch installed

you might need to make it match your cuda version etc..

We only support torch>=2.0.0

then install laminDB

```bash
pip install 'lamindb[jupyter,bionty]'
```

then install scPrint

```bash
pip install scprint
```

> I had to install a specific version of pytorch, torchaudio, torchtext.. for my cuda version.
My cuda compiler nvcc compiles cuda 11.7. my cuda-smi (api) is currently 12.1.

> Please install all of it for your cuda version and it should still work. for more information on this, please see [installation.md](installation.md).

### in dev mode

```python
conda create ...
git clone https://github.com/jkobject/scPRINT
cd scPRINT
git checkout dev
git submodule init
git submodule update
# install pytorch as mentionned above if you have a GPU
pip install -e .[dev]
# install triton as mentioned in .toml if you want to
mkdocs serve # to view the dev documentation
```

## Usage

```py
from lightning.pytorch import Trainer
from scprint import scPrint
from scdataloader import DataModule

...
model = scPrint(...)
trainer = Trainer(...)
trainer.fit(model, datamodule=datamodule)
```

```bash
$ python -m scPrint/__main__.py
#or
$ scprint fit/train/predict/test
```

for more information on usage please see the documentation in https://jkobject.com/scPrint

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

### What is included?

- 📃 Documentation structure using [mkdocs](http://www.mkdocs.org)
- 🧪 Testing structure using [pytest](https://docs.pytest.org/en/latest/)
  If you want [codecov](https://about.codecov.io/sign-up/) Reports and Automatic Release to [PyPI](https://pypi.org)  
  On the new repository `settings->secrets` add your `PYPI_API_TOKEN` and `CODECOV_TOKEN` (get the tokens on respective websites)
- ✅ Code linting using [flake8](https://flake8.pycqa.org/en/latest/)
- 📊 Code coverage reports using [codecov](https://about.codecov.io/sign-up/)
- 🛳️ Automatic release to [PyPI](https://pypi.org) using [twine](https://twine.readthedocs.io/en/latest/) and github actions.


acknowledgement:
[python template](https://github.com/rochacbruno/python-project-template)
[scGPT]()
[laminDB]()
