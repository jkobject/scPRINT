import scanpy as sc
import numpy as np
import os
import urllib.request

from scprint.base import NAME
from scdataloader import Preprocessor
from scdataloader.utils import populate_my_ontology
import subprocess


def test_base():
    assert NAME == "scprint"
    populate_my_ontology(
        organisms=["NCBITaxon:10090", "NCBITaxon:9606"],
        sex=["PATO:0000384", "PATO:0000383"],
        celltypes=None,
        ethnicities=None,
        assays=None,
        tissues=None,
        diseases=None,
        dev_stages=None,
    )
    filepath = os.path.join(os.path.dirname(__file__), "test.h5ad")
    ckpt_path = os.path.join(os.path.dirname(__file__), "small.ckpt")
    if not os.path.exists(ckpt_path):
        url = "https://huggingface.co/jkobject/scPRINT/resolve/main/small.ckpt"
        urllib.request.urlretrieve(url, ckpt_path)

    result = subprocess.run(
        [
            "scprint",
            "denoise",
            "--ckpt_path",
            ckpt_path,
            "--adata",
            filepath,
            "--output_filename",
            "out",
            "--plot_corr_size",
            "10",
            "--species",
            "NCBITaxon:9606",
        ],
        capture_output=True,
        text=True,
    )

    exit_code = result.returncode
    print("stdout:", result.stdout)
    if exit_code != 0:
        print("stderr:", result.stderr)
        assert False, f"scPRINT denoise test failed with exit code {exit_code}"
    assert exit_code == 0, f"scPRINT denoise test failed with exit code {exit_code}"
