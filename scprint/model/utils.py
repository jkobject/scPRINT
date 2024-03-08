import torch
import numpy as np
from typing import Optional, Union
from torch.distributions import Poisson, Gamma
import bionty as bt
from collections import Counter
import math
import torch.nn as nn

import scanpy as sc
from anndata import AnnData

import numpy as np
from matplotlib import pyplot as plt


def make_adata(
    pred, embs, labels, step=0, label_decoders=None, gtclass=None, name="", mdir="/tmp"
):
    colname = ["pred_" + i for i in labels]
    obs = np.array(pred.to(device="cpu", dtype=torch.int32))
    # label decoders is not cls_decoders. one is a dict to map class codes (ints)
    # to class names the other is the module the predict the class
    if label_decoders is not None:
        obs = np.array(
            [
                [label_decoders[labels[i]][n] for n in name]
                for i, name in enumerate(obs.T)
            ]
        ).T

    if gtclass is not None:
        colname += labels
        nobs = np.array(gtclass.to(device="cpu", dtype=torch.int32))
        if label_decoders is not None:
            nobs = np.array(
                [
                    [label_decoders[labels[i]][n] for n in name]
                    for i, name in enumerate(nobs.T)
                ]
            ).T
        obs = np.hstack([obs, nobs])

    adata = AnnData(
        np.array(embs.to(device="cpu", dtype=torch.float32)),
        obs=pd.DataFrame(
            obs,
            columns=colname,
        ),
    )

    for n in labels:
        if gtclass is not None:
            tr = translate(adata.obs[n].tolist(), n)
            if tr is not None:
                adata.obs["conv_" + n] = adata.obs[n].replace(tr)
        tr = translate(adata.obs["pred_" + n].tolist(), n)
        if tr is not None:
            adata.obs["conv_pred_" + n] = adata.obs["pred_" + n].replace(tr)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)
    adata.obs = adata.obs.astype("category")
    print(adata)
    if gtclass is not None:
        color = [
            i
            for pair in zip(
                [
                    "conv_" + i if "conv_" + i in adata.obs.columns else i
                    for i in labels
                ],
                [
                    "conv_pred_" + i
                    if "conv_pred_" + i in adata.obs.columns
                    else "pred_" + i
                    for i in labels
                ],
            )
            for i in pair
        ]
        fig, axs = plt.subplots(int(len(color) / 2), 2, figsize=(24, len(color) * 4))
        plt.subplots_adjust(wspace=1)
        for i, col in enumerate(color):
            sc.pl.umap(
                adata,
                color=col,
                ax=axs[i // 2, i % 2],
                show=False,
            )
    else:
        color = [
            "conv_pred_" + i if "conv_pred_" + i in adata.obs.columns else "pred_" + i
            for i in labels
        ]
        fig, axs = plt.subplots(len(color), 1, figsize=(16, len(color) * 8))
        for i, col in enumerate(color):
            sc.pl.umap(
                adata,
                color=col,
                ax=axs[i],
                show=False,
            )
    adata.write(mdir + "/step_" + str(step) + "_" + name + ".h5ad")
    return adata


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    mup_width_scale=1.0,
    rescale_prenorm_residual=True,
):
    mup_init_scale = math.sqrt(mup_width_scale)
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range * mup_init_scale)
        optim_cfg = getattr(module.weight, "_optim", {})
        optim_cfg.update({"lr_multiplier": mup_width_scale})
        setattr(module.weight, "_optim", optim_cfg)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        pass

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(
                    p,
                    mean=0.0,
                    std=initializer_range * mup_init_scale / math.sqrt(2 * n_layer),
                )


def downsample_profile(mat, renoise):
    """
    This function downsamples the expression profile of a given single cell RNA matrix.

    The noise is applied based on the renoise parameter,
    the total counts of the matrix, and the number of genes. The function first calculates the noise
    threshold (tnoise) based on the renoise parameter. It then generates an initial matrix count by
    applying a Poisson distribution to a random tensor scaled by the total counts and the number of genes.
    The function then models the sampling zeros by applying a Poisson distribution to a random tensor
    scaled by the noise threshold, the total counts, and the number of genes. The function also models
    the technical zeros by generating a random tensor and comparing it to the noise threshold. The final
    matrix count is calculated by subtracting the sampling zeros from the initial matrix count and
    multiplying by the technical zeros. The function ensures that the final matrix count is not less
    than zero by taking the maximum of the final matrix count and a tensor of zeros. The function
    returns the final matrix count.

    Args:
        mat (torch.Tensor): The input matrix.
        renoise (float): The renoise parameter.
        totcounts (torch.Tensor): The total counts of the matrix.
        ngenes (int): The number of genes.

    Returns:
        torch.Tensor: The matrix count after applying noise.
    """
    # Randomly drop on average N counts to each element of expression using a heavy tail Gaussian distribution
    # here we try to get the scale of the distribution so as to remove the right number of counts from each gene
    # https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02601-5#:~:text=Zero%20measurements%20in%20scRNA%2Dseq,generation%20of%20scRNA%2Dseq%20data.
    totcounts = mat.sum(1)
    batch = mat.shape[0]
    ngenes = mat.shape[1]
    tnoise = 1 - (1 - renoise) ** (1 / 2)
    # we model the sampling zeros (dropping 30% of the reads)
    res = torch.poisson(
        torch.rand((batch, ngenes)).to(device=mat.device)
        * ((tnoise * totcounts.unsqueeze(1)) / (0.5 * ngenes))
    ).int()
    # we model the technical zeros (dropping 50% of the genes)
    drop = (torch.rand((batch, ngenes)) > tnoise).int().to(device=mat.device)

    mat = (mat - res) * drop
    return torch.maximum(mat, torch.Tensor([[0]]).to(device=mat.device)).int()


def masker(
    length: int,
    batch_size: int = 1,
    mask_ratio: float = 0.15,
    mask_prob: Optional[Union[torch.Tensor, np.ndarray]] = None,  # n_features
    mask_value: int = 1,
) -> torch.Tensor:
    """
    Randomly mask a batch of data.

    Args:
        values (array-like):
            A batch of tokenized data, with shape (batch_size, n_features).
        mask_ratio (float): The ratio of genes to mask, default to 0.15.
        mask_value (int): The value to mask with, default to -1.
        pad_value (int): The value of padding in the values, will be kept unchanged.

    Returns:
        torch.Tensor: A tensor of masked data.
    """
    mask = []
    for _ in range(batch_size):
        m = np.zeros(length)
        loc = np.random.choice(
            a=length, size=int(length * mask_ratio), replace=False, p=mask_prob
        )
        m[loc] = mask_value
        mask.append(m)

    return torch.Tensor(np.array(mask)).to(torch.bool)


def zinb_sample(mu, theta, zi_probs, sample_shape=torch.Size([])):
    concentration = theta
    rate = theta / mu
    # Important remark: Gamma is parametrized by the rate = 1/scale!
    gamma_d = Gamma(concentration=concentration, rate=rate)
    p_means = gamma_d.sample(sample_shape)

    # Clamping as distributions objects can have buggy behaviors when
    # their parameters are too high
    l_train = torch.clamp(p_means, max=1e8)
    samp = Poisson(l_train).sample()  # Shape : (n_samples, n_cells_batch, n_vars)
    is_zero = torch.rand_like(samp) <= zi_probs
    samp_ = torch.where(is_zero, torch.zeros_like(samp), samp)
    return samp_


def translate(val, t="cell_type_ontology_term_id"):
    if t == "cell_type_ontology_term_id":
        obj = bt.CellType.df().set_index("ontology_id")
    elif t == "assay_ontology_term_id":
        obj = bt.ExperimentalFactor.df().set_index("ontology_id")
    elif t == "tissue_ontology_term_id":
        obj = bt.Tissue.df().set_index("ontology_id")
    elif t == "disease_ontology_term_id":
        obj = bt.Disease.df().set_index("ontology_id")
    elif t == "self_reported_ethnicity_ontology_term_id":
        obj = bt.Ethnicity.df().set_index("ontology_id")
    else:
        return None
    if type(val) is str:
        return {val: obj.loc[val]["name"]}
    elif type(val) is list or type(val) is set:
        return {i: obj.loc[i]["name"] for i in set(val)}
    elif type(val) is dict or type(val) is Counter:
        return {obj.loc[k]["name"]: v for k, v in val.items()}
