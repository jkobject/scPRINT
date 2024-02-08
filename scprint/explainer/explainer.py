# from .logger import *
# from .visualization import *
import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
from bertviz import head_view
from grnndata import GRNAnnData
from torch.nn.functional import softmax
from welford_torch import Welford

from scgpt.tokenizer import tokenize_and_pad_batch

import gseapy as gp
from gseapy.plot import dotplot

##TEMP##
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
########
from attention_flow import *


### @title Explainable Attention ####
class BaseExplainer:
    def __init__(self, model, dataset) -> None:
        self.model = model
        self.dataset = dataset

    def get_attention(self, layer=None, save_path=""):
        ### run the model with the data
        # model_output = self.model.encode(self.dataset)
        all_hidden_states, all_attentions = (
            model_output["hidden_states"],
            model_output["attentions"],
        )
        _attentions = [att.detach().numpy() for att in all_attentions]
        attentions_mat = np.asarray(_attentions)[:, 0]
        if len(save_path) > 0:
            # save as pkl
            with open(save_path + "_attn_scores.pkl", "wb") as f:
                pickle.dump(attn_scores, f)
        ## TODO: show highly attended genes:

        return attn_scores

    def bertview(
        self, loc=(0, 10), key_genes=[], gene_names=[], random=False, **kwargs
    ):
        attn, var_attn, _ = self.get_attention(**kwargs)
        for i in range(0, attn.shape[0]):
            attn[i][(var_attn[i] + 0.00001) / (0.00001 + attn[i]) > 0.15] = 0
        if len(loc) > 0:
            todisp = torch.FloatTensor(
                [attn[:, loc[0] : loc[1], :][:, : loc[0] : loc[1]]]
            )

            names = self.dataset.var["feature_name"].values[loc[0] : loc[1]]
        elif len(key_genes) > 0:
            names = key_genes
        elif random:
            random_indices = np.random.randint(low=0, high=attn.shape[1], size=80)
            todisp = torch.FloatTensor(
                np.expand_dims(attn[:, random_indices, :][:, :, random_indices], axis=0)
            )
            names = self.dataset.var["feature_name"].values[random_indices]
        else:
            raise ValueError(
                "Please provide either a location or a list of genes to visualize"
            )
        if todisp.max() < 1:
            print("re-scaling todisp so that we can see something")
            todisp = todisp / todisp.max()

        head_view([todisp], names)

    def viz_gene_embeddings(self, adata=None, layer=None, resolutions=[0.5, 1, 4]):
        if layer is not None:
            self.get_gene_embeddings(layer=layer)
        elif adata is None:
            raise ValueError("Please provide either adata or layer")
        sc.pp.pca(adata, n_comps=50)
        sc.pp.neighbors(adata, n_pcs=50, n_neighbors=20)
        sc.tl.umap(adata, min_dist=0.3)

        [
            sc.tl.leiden(adata, resolution=0.5, key_added="leiden_res" + str(res))
            for res in resolutions
        ]
        sc.pl.umap(adata, color=["leiden_res" + str(res) for res in resolutions])

    def extract_grn(self, layer, resolution=0.5, n_neighbors=20):
        att, var_attn, genes = self.get_attention(layer=layer)
        for i in range(0, att.shape[0]):
            att[i][var_attn[i] > att / 10] = 0
        grn = np.mean(att, axis=0)
        return GRNAnnData(adata=self.dataset.to_adata, grn=grn)

    # TODO: get the marker genes of a specific class or set of classes using something like LRP

    # TODO: get the effect of a knock out / knock in


class GeneFormer_Explainer(BaseExplainer):
    def get_gene_embeddings(
        self, pickle_file="../geneformer/token_dictionary.pkl", layer=0
    ) -> ad.AnnData:
        assert layer == 0, "GeneFormer only has one layer"
        di = pd.read_pickle(pickle_file)
        val = di.keys()
        adata = sc.AnnData(
            pd.DataFrame(
                index=list(val),
                data=self.model.embeddings.word_embeddings.weight.data.cpu(),
            )
        )
        return adata


class scGPT_Explainer(BaseExplainer):
    def __init__(self, *args, vocab, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_value = -2
        self.pad_token = "<pad>"
        self.vocab = vocab

    def get_attention(self, layer_num=11, batch_size=8, as_df=False):
        torch.cuda.empty_cache()
        # dict_sum_condition = {}

        tokenized_all = tokenize_and_pad_batch(
            self.dataset.X.toarray(),
            self.dataset.var["gene_ids"].tolist(),
            max_len=self.dataset.shape[1] + 1,
            vocab=self.vocab,
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=True,
        )
        all_gene_ids, all_values = (
            tokenized_all["genes"],
            tokenized_all["values"],
        )
        src_key_padding_mask = all_gene_ids.eq(self.vocab[self.pad_token])
        self.model.eval()
        w = Welford()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            N = all_gene_ids.size(0)
            device = next(self.model.parameters()).device

            for i in tqdm(range(0, N, batch_size)):
                batch_size = all_gene_ids[i : i + batch_size].size(0)
                # Replicate the operations in self.model forward pass
                src_embs = self.model.encoder(
                    torch.tensor(all_gene_ids[i : i + batch_size], dtype=torch.long).to(
                        device
                    )
                )
                val_embs = self.model.value_encoder(
                    torch.tensor(all_values[i : i + batch_size], dtype=torch.float).to(
                        device
                    )
                )
                total_embs = src_embs + val_embs
                # total_embs = self.model.layer(total_embs.permute(0, 2, 1)).permute(0, 2, 1)
                # Send total_embs to attention layers for attention operations
                # Retrieve the output from second to last layer
                for layer in self.model.transformer_encoder.layers[:layer_num]:
                    total_embs = layer(
                        total_embs,
                        src_key_padding_mask=src_key_padding_mask[
                            i : i + batch_size
                        ].to(device),
                    )
                # Send total_embs to the last layer in flash-attn
                # https://github.com/HazyResearch/flash-attention/blob/1b18f1b7a133c20904c096b8b222a0916e1b3d37/flash_attn/flash_attention.py#L90
                qkv = self.model.transformer_encoder.layers[layer_num].self_attn.Wqkv(
                    total_embs
                )
                # Retrieve q, k, and v from flast-attn wrapper
                qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=8)
                q = qkv[:, :, 0, :, :]
                k = qkv[:, :, 1, :, :]
                # v = qkv[:, :, 2, :, :]
                # https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a
                # q = [batch, gene, n_heads, n_hid]
                # k = [batch, gene, n_heads, n_hid]
                # attn_scores = [batch, n_heads, gene, gene]
                attn_scores = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)
                # apply softmax to get attention weights
                attn_scores = softmax(attn_scores, dim=-1)
                [w.add(attn_score) for attn_score in attn_scores]
        gene_vocab_idx = all_gene_ids[0].clone().detach().cpu().numpy()
        if as_df:
            sm_attn_scores = [
                pd.DataFrame(
                    data=w.mean[i].clone().detach().cpu().numpy(),
                    columns=self.vocab.lookup_tokens(gene_vocab_idx),
                    index=self.vocab.lookup_tokens(gene_vocab_idx),
                )
                for i in range(0, 8)
            ]
            var_attn_scores = [
                pd.DataFrame(
                    data=w.var_s[i].clone().detach().cpu().numpy(),
                    columns=self.vocab.lookup_tokens(gene_vocab_idx),
                    index=self.vocab.lookup_tokens(gene_vocab_idx),
                )
                for i in range(0, 8)
            ]
            return sm_attn_scores, var_attn_scores, gene_vocab_idx
        else:
            return (
                w.mean.clone().detach().cpu().numpy(),
                w.var_s.clone().detach().cpu().numpy(),
                gene_vocab_idx,
            )

    # return [pd.DataFrame(data=sm_attn_scores[i], columns=vocab.lookup_tokens(gene_vocab_idx), index=vocab.lookup_tokens(gene_vocab_idx)) for i in range(0,8)]
