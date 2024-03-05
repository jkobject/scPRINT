# from scprint.base.base_model import BaseModel

from typing import Optional, Dict
from torch import Tensor, optim, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from lightning.pytorch.tuner.lr_finder import _LRCallback
from lightning.pytorch.callbacks.lr_finder import LearningRateFinder
import torch.distributed as dist
import torch

import lightning as L

import pandas as pd
import scanpy as sc
from anndata import AnnData
import math
from functools import partial
import numpy as np
from matplotlib import pyplot as plt

try:
    from .flash_attn import FlashTransformerEncoder
    from .hashformer import Hashformer
except ModuleNotFoundError as e:
    print(e)
    print(
        "can't use flash attention and triton kernel,\
        you likely don't have the right hardware or didn't \
        make the right installation"
    )
    MHA = None
    Block = None
    Hashformer = None

from . import encoders
from . import decoders

# from .linear_transformer import FastTransformerEncoderWrapper as FastTransformerEncoder
from .EGT import EGT
from .dsbn import DomainSpecificBatchNorm1d
from . import loss
from .utils import masker
from . import utils

import time


class scPrint(L.LightningModule):
    def __init__(
        self,
        genes: list,
        precpt_gene_emb: Optional[str] = None,
        gene_pos_enc: Optional[list] = None,
        d_model: int = 512,
        nhead: int = 8,
        d_hid: int = 512,
        edge_dim: int = 12,
        nlayers: int = 6,
        layers_cls: list[int] = [],
        labels: Dict[str, int] = {},
        cls_hierarchy: Dict[str, Dict[int, list[int]]] = {},
        dropout: float = 0.2,
        transformer: str = "fast",
        expr_emb_style: str = "continuous",  # "binned_pos", "cont_pos"
        domain_spec_batchnorm: str = "None",
        n_input_bins: int = 0,
        mvc_decoder: str = "inner product",
        pred_embedding: list[str] = [],
        cell_emb_style: str = "cls",
        lr=0.001,
        label_decoders: Optional[Dict[str, Dict[int, str]]] = None,
        strict_loading: bool = True,
        **flash_attention_kwargs,
    ):
        """
        __init__ method for TransformerModel.
        # TODO: add docstrings
        Args:
            genes (list): the genenames with which the model will work
            precpt_gene_emb (np.array, optional): The gene embeddings. should be of size len(genes), d_model.
                it should be in the same order as the genes. Defaults to None.
            gene_pos_enc (list, optional): The gene position encoding. Should be of the same size as genes.
                for each gene in genes, gives it a location value. Defaults to None.
            d_model (int, optional): The dimension of the model. Defaults to 512.
            nhead (int, optional): The number of heads in the multiheadattention models. Defaults to 8.
            d_hid (int, optional): The dimension of the feedforward network model. Defaults to 512.
            nlayers (int, optional): The number of layers in the transformer model. Defaults to 6.
            nlayers_cls (int, optional): The number of layers in the classifier. Defaults to 3.
            n_cls (int, optional): The number of classes. Defaults to 0.
            dropout (float, optional): The dropout value. Defaults to 0.5.
            do_adv (bool, optional): Whether to perform adversarial discrimination. Defaults to False.
            domain_spec_batchnorm str], optional): Whether to apply domain specific batch normalization. Defaults to False.
            expr_emb_style (str, optional): The style of input embedding (one of "continuous_concat", "binned_pos", "full_pos"). Defaults to "continuous_concat".
            mvc_decoder (str, optional): The style of MVC decoder one of "None", "inner product", "concat query", "sum query". Defaults to "inner product".
            ecs_threshold (float, optional): The threshold for the cell similarity. Defaults to 0.3.
            pre_norm (bool, optional): Whether to apply pre normalization. Defaults to False.

        Raises:
            ValueError: If the expr_emb_style is not one of "category", "continuous", "none".
        """
        super().__init__()
        # default
        self.do_denoise = False
        self.noise = []
        self.do_cce = True
        self.cce_sim = 0.5
        self.do_ecs = True
        self.ecs_threshold = 0.3
        self.ecs_scale = 1.0
        self.do_mvc = False
        self.do_adv_cls = False
        self.do_next_tp = False
        self.do_generate = False
        self.class_scale = 1000
        self.mask_ratio = [0.15]
        self.warmup_duration = 500
        self.weight_decay = 0.0
        self.fused_adam = False
        self.lr_patience = 3
        self.lrfinder_steps = 0
        self.embs = None
        # should be stored somehow
        self.d_model = d_model
        self.edge_dim = edge_dim
        self.nlayers = nlayers
        self.gene_pos_enc = gene_pos_enc
        self.mvc_decoder = mvc_decoder
        self.domain_spec_batchnorm = domain_spec_batchnorm
        # need to store
        self.n_input_bins = n_input_bins
        self.transformer = transformer
        self.labels_counts = labels
        self.labels = list(labels.keys())
        self.cell_emb_style = cell_emb_style
        self.label_decoders = label_decoders
        self.pred_embedding = pred_embedding
        self.lr = lr
        self.strict_loading = strict_loading
        # compute tensor for cls_hierarchy
        self.cls_hierarchy = {}

        for k, v in cls_hierarchy.items():
            tens = torch.zeros((len(v), labels[k]))
            for k2, v2 in v.items():
                tens[k2 - labels[k], v2] = 1
            self.cls_hierarchy[k] = tens.to(bool)
        self.expr_emb_style = expr_emb_style

        if self.expr_emb_style not in ["category", "continuous", "none"]:
            raise ValueError(
                f"expr_emb_style should be one of category, continuous, scaling, "
                f"got {expr_emb_style}"
            )
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")

        self.genes = genes
        self.vocab = {i: n for i, n in enumerate(genes)}

        # encoder
        # gene encoder
        if precpt_gene_emb is not None:
            embeddings = pd.read_parquet(precpt_gene_emb).loc[self.genes]
            if len(embeddings) == 0:
                raise ValueError(
                    f"the gene embeddings file {precpt_gene_emb} does not contain any of the genes given to the model"
                )
            elif len(embeddings) < len(self.genes):
                print(
                    "Warning: only a subset of the genes available in the embeddings file."
                )
                print("number of genes: ", len(embeddings))
            sembeddings = torch.nn.AdaptiveAvgPool1d(d_model)(
                torch.tensor(embeddings.values)
            )
            self.gene_encoder = encoders.GeneEncoder(
                len(self.vocab), d_model, weights=sembeddings, freeze=True
            )
        else:
            self.gene_encoder = encoders.GeneEncoder(len(self.vocab), d_model)

        # Value Encoder, NOTE: the scaling style is also handled in _encode method
        if expr_emb_style in ["continuous", "full_pos"]:
            self.expr_encoder = encoders.ContinuousValueEncoder(d_model, dropout)
        elif expr_emb_style == "binned_pos":
            assert n_input_bins > 0
            self.expr_encoder = encoders.CategoryValueEncoder(n_input_bins, d_model)
        else:
            self.expr_encoder = nn.Identity()

        # Positional Encoding
        if self.gene_pos_enc is not None:
            max_len = max(gene_pos_enc)
            token_to_pos = {token: pos for token, pos in enumerate(self.gene_pos_enc)}
            self.pos_encoder = encoders.PositionalEncoding(
                d_model, max_len=max_len, token_to_pos=token_to_pos
            )

        # Batch Encoder
        # always have [base_cell_emb, time_embedding, depth_embedding] + any other class info
        # base cell embedding will store other cell specific information
        self.label_encoder = encoders.CategoryValueEncoder(
            len(self.labels) + 2, d_model
        )
        # self.time_encoder = encoders.ContinuousValueEncoder(d_model, dropout)
        self.depth_coder = encoders.ContinuousValueEncoder(d_model, dropout)

        # Model
        # Batch Norm
        if domain_spec_batchnorm is True or domain_spec_batchnorm == "dsbn":
            use_affine = True if domain_spec_batchnorm == "do_affine" else False
            print(f"Use domain specific batchnorm with affine={use_affine}")
            self.dsbn = DomainSpecificBatchNorm1d(
                d_model, len(self.labels), eps=6.1e-5, affine=use_affine
            )
        elif domain_spec_batchnorm == "batchnorm":
            print("Using simple batchnorm instead of domain specific batchnorm")
            self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)

        # Transformer
        # Linear
        if transformer == "linear":
            # linear transformer using the fast transformer package
            self.transformer = FastTransformerEncoder(
                d_model, nhead, d_hid, nlayers, dropout, "linear"
            )
        # flash
        elif transformer == "flash":
            if FlashTransformerEncoder is None:
                raise ValueError("flash transformer requires flash package")
            # NOT flash transformer using the special tritton kernel
            # or parallelMHA (add the process group thing and faster)
            self.transformer = FlashTransformerEncoder(
                d_model, nhead, nlayers, dropout=dropout, **flash_attention_kwargs
            )

        # flashsparse
        elif transformer == "flashsparse":
            if Hashformer is None:
                raise ValueError("Hashformer transformer requires cuda kernels")
            self.transformer = Hashformer(
                d_model,
                nlayers,
                2,
                nhead,
            )
        # flash EGT
        # We found that the results can be further improved by freezing the
        # node channel layers and training the edge channel layers for a
        # few additional epochs.
        # However, its effect on transfer learning has not yet been studied.
        # That is why we include checkpoints for both tuned and untuned models.
        # https://github.com/shamim-hussain/egt/blob/master/README.md
        # https://github.com/shamim-hussain/egt_pytorch

        elif transformer == "scprint":
            self.transformer = EGT(
                num_layers=nlayers,
                feat_size=d_model,
                edge_feat_size=edge_dim,
                num_heads=nhead,
                num_virtual_nodes=len(self.labels),
            )
        # regular
        else:
            self.transformer = TransformerEncoder(
                TransformerEncoderLayer(
                    d_model, nhead, d_hid, dropout, batch_first=True
                ),
                nlayers,
            )

        # decoders
        # expression
        self.expr_decoder = decoders.ExprDecoder(
            d_model,
            nfirst_labels_to_skip=len(self.labels) + 2,
            dropout=dropout,
        )
        # cls decoder
        self.cls_decoders = nn.ModuleDict()
        # should be a very simple classifier for most things
        # (maybe scale with the number of classes) should be 1 layer...
        for label, n_cls in labels.items():
            self.cls_decoders[label] = decoders.ClsDecoder(
                d_model, n_cls, layers=layers_cls, dropout=dropout
            )

        # expression decoder from batch embbedding
        if mvc_decoder is not None:
            self.mvc_decoder = decoders.MVCDecoder(
                d_model,
                arch_style=mvc_decoder,
                dropout=dropout,
            )
        else:
            self.mvc_decoder = None

        self.apply(
            partial(
                _init_weights,
                n_layer=nlayers,
                # initializer_range=initializer_range,
                # mup_width_scale=getattr(config, "mup_width_scale", 1.0),
            )
        )
        self.save_hyperparameters()
        print(self)

    def _encoder(
        self,
        gene_pos: Tensor,
        expression: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        full_depth: Optional[Tensor] = None,
        # (minibatch,) unormalized total counts
        # (minibatch,), then will be compared to this timepoint
        timepoint: Optional[Tensor] = None,
        cell_embs: Optional[Tensor] = None,  # (minibatch, n_labels, embsize)
        get_attention: bool = False,
        attention_layer: Optional[int] = None,
    ):
        """
        _encode given gene expression, encode the gene embedding and cell embedding.

        Args:
            gene_pos (Tensor): _description_
            expression (Tensor): _description_
            mask (Tensor): boolean, same size as gene_pos and has 1 for masked expression locations
            labels (Optional[Tensor], optional): _description_. Defaults to None.

        Returns:
            Tensor: _description_
        """
        enc = self.gene_encoder(gene_pos)  # (minibatch, seq_len, embsize)
        self.cur_gene_token_embs = enc.clone()
        if expression is not None:
            enc += self.expr_encoder(
                torch.log2(1 + expression), mask
            )  # (minibatch, seq_len, embsize)

        if self.gene_pos_enc:
            enc += self.pos_encoder(gene_pos)
        # else:
        #    total_embs = torch.cat([genc, exp_enc], dim=-1)
        # if self.expr_emb_style == "scaling":
        #    exp_enc = exp_enc.unsqueeze(2)
        #    total_embs = genc * exp_enc
        # else:
        # if cell embedding is already provided, we don't compute the default ones
        cell_embs = (
            self.label_encoder(
                torch.Tensor([list(range(len(self.labels) + 2))] * gene_pos.shape[0])
                .int()
                .to(gene_pos.device)
            )
            if cell_embs is None
            else cell_embs
        )  # (minibatch, embsize)
        if timepoint is not None:
            pass
            # cell_embs[:, 2, :] = self.time_encoder(timepoint)
        if full_depth is not None:
            cell_embs[:, 1, :] = self.depth_encoder(torch.log2(1 + full_depth))

        enc = torch.cat([cell_embs, enc], dim=1)

        # TODO: seems to be a problem here:
        # if getattr(self, "dsbn", None) is not None and batch_label is not None:
        #     label = int(labels[0].item())
        #     total_embs = self.dsbn(total_embs.permute(0, 2, 1), label).permute(
        #         0, 2, 1
        #     )  # the batch norm always works on dim 1
        # elif getattr(self, "bn", None) is not None:
        #     total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.transformer(enc)
        # TODO: get the attention here
        return output  # (minibatch, seq_len, embsize)

    def _decoder(
        self, transformer_output, depth_mult, get_gene_emb=False, do_sample=False
    ):
        output = self.expr_decoder(transformer_output)

        output["mean"] = depth_mult.unsqueeze(1) * output["mean"]
        if do_sample:
            pass
        # bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
        # output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]

        output["cell_embs"] = self._get_cell_embs(transformer_output)
        cell_emb = torch.mean(output["cell_embs"], dim=1)
        output["cell_emb"] = cell_emb  # batch * d_model
        if len(self.labels) > 0:
            output.update(
                {
                    "cls_output_"
                    + labelname: self.cls_decoders[labelname](
                        output["cell_embs"][
                            :, 2 + i, :
                        ]  # the first elem is the base cell embedding
                    )
                    for i, labelname in enumerate(self.labels)
                }
            )  # (minibatch, n_cls)
        if self.mvc_decoder is not None:
            output.update(self.mvc_decoder(cell_emb, self.cur_gene_token_embs))
            output["mvc_mean"] = (
                depth_mult.unsqueeze(1) * output["mvc_mean"]
            )  # (minibatch, seq_len)

        # if self.do_adv:
        # TODO: do DAB
        # output["dab_output"] = self.grad_reverse_discriminator(cell_emb)

        if get_gene_emb:
            output["gene_embedding"] = transformer_output[
                :, len(self.labels) + 2 :, :
            ]  # (minibatch, seq_len, embsize)
        return output

    def forward(
        self,
        gene_pos: Tensor,
        depth_mult: Tensor,
        expression: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        # (minibatch,) unormalized total counts
        full_depth: Optional[Tensor] = None,
        timepoint: Optional[Tensor] = None,  # (new_minibatch_of_nxt_cells,)
        get_gene_emb: bool = False,
        do_sample: bool = False,
    ):
        """
        Args:
            gene_pos (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            expression (:obj:`Tensor`): token values, shape [batch_size, seq_len]

        Returns:
            dict of output Tensors.
        """
        transformer_output = self._encoder(
            gene_pos, expression, mask, full_depth, timepoint
        )
        return self._decoder(transformer_output, depth_mult, get_gene_emb, do_sample)

    def configure_optimizers(self):
        # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        # optimizer = optim.Adam(
        #    self.parameters(),
        #    lr=self.hparams.lr,
        #    betas=(0.9, 0.999),
        #    eps=1e-08,
        #    weight_decay=0,
        #    amsgrad=False,
        #    fused=False,
        # )
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.weight_decay,
            amsgrad=False,
            fused=self.fused_adam,
        )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=self.lr_patience, factor=0.5
        )
        lr_dict = {
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss" if self.trainer.val_dataloaders else "train_loss",
        }
        self.lrfinder_steps = 0
        for val in self.trainer.callbacks:
            if type(val) is _LRCallback:
                self.lrfinder_steps = val.num_training
            if type(val) is LearningRateFinder:
                self.lrfinder_steps = val._num_training_steps
        return [optimizer], [lr_dict]

    def on_fit_start(self):
        if type(self.transformer) is FlashTransformerEncoder:
            for encoder_layers in self.transformer.blocks:
                encoder_layers.set_seq_parallel(True)
        for k, v in self.cls_hierarchy.items():
            self.cls_hierarchy[k] = v.to(self.device)

    def training_step(
        self,
        batch,
        batch_idx,
    ):
        """
        training_step defines the train loop. It is independent of forward

        Args:
            batch (list[Tensor]): shape [Tensor(minibatch, seq_len)*2, Tensor(minibatch, n_labels)]
                the n_labels should be in the same orders as the labels provided in the model
                the first Tensor is gene ids, the second is expression of those gene_pos
            batch_idx (Tensor): shape (minibatch)
            superbatch (list[Tensor], optional): shape [(neighbors, seq_len)*minibatch | None]
                gives out additional expression (on the same gene_pos) for the k
                nearest neighbors of the cell at the same minibatch index in "batch"). Defaults to None.
            superbatch_idx (Tensor, optional): _description_. Defaults to None.
            do_cce (:obj:`bool`): if True, return the contrastive cell embedding objective
                (CCE) output
            do_ecs (:obj:`bool`): if True, return the elastic cell similarity objective
                (ECS) output.
            mask_ratio (list[float], optional): the ratio of masked gene_pos to use for the MLM tasks
                the first mask ratio will be used for the default embedding learning


        Returns:
            _type_: _description_
        """
        # TASK 1 & 2 & 3 (first pass, expression reconstruction, label prediction)
        total_loss, losses = self._full_training(
            batch,
            self.do_denoise,
            self.noise,
            self.do_next_tp,
            self.do_cce,
            self.cce_sim,
            self.do_ecs,
            self.do_mvc,
            self.do_adv_cls,
            self.do_generate,
            self.mask_ratio,
        )
        self.log("train_loss", total_loss, prog_bar=True)
        self.log_dict(losses, prog_bar=True)
        return total_loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        # making sure that we don't do this during lrfinder
        if (
            self.trainer.global_step < self.warmup_duration + self.lrfinder_steps
        ) and self.lrfinder_steps < self.trainer.global_step:
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.warmup_duration
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
            self.log("lr", lr_scale * self.hparams.lr)
        else:
            self.log("lr", self.lr)

    def _full_training(
        self,
        batch,
        do_denoise=False,
        noise=[],
        do_next_tp=False,
        do_cce=False,
        cce_sim=0.5,
        do_ecs=False,
        do_mvc=False,
        do_adv_cls=False,
        do_generate=False,
        mask_ratio=[0.15],
    ):
        if type(mask_ratio) is not list:
            mask_ratio = [mask_ratio]

        expression = batch["x"]
        gene_pos = batch["genes"]
        clss = batch["class"]
        total_count = batch["depth"]
        timepoint = None

        total_loss = 0
        losses = {}
        cell_embs = []
        default_embs = None
        # depth = torch.min(
        #    torch.tensor(1),
        #    torch.log2(torch.max(total_count, torch.tensor(100)) / 100) / 19,
        # ).to(gene_pos.device)
        for i in mask_ratio:
            mask = masker(
                length=gene_pos.shape[1],
                batch_size=gene_pos.shape[0],
                mask_ratio=i,
            ).to(gene_pos.device)
            output = self.forward(
                gene_pos, expression.sum(1), expression, mask, total_count, timepoint
            )
            l, tot = self._compute_loss(
                output, expression, mask, clss, do_ecs, do_adv_cls, do_mvc
            )

            cell_embs.append(output["cell_emb"].clone())
            if default_embs is None:
                default_embs = output["cell_embs"].clone()
            total_loss += tot
            losses.update(
                {"mask_" + str(int(i * 100)) + "%_" + k: v for k, v in l.items()}
            )
        # TASK 3. denoising
        if do_denoise:
            for i in noise:
                # Randomly drop on average N counts to each element of expression using a heavy tail Gaussian distribution
                # here we try to get the scale of the distribution so as to remove the right number of counts from each gene
                # https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02601-5#:~:text=Zero%20measurements%20in%20scRNA%2Dseq,generation%20of%20scRNA%2Dseq%20data.
                # TODO: test and look a bit more into it
                expr = utils.downsample_profile(expression, renoise=i)
                true_rescale = expr.sum(1) / expression.sum(1)
                output = self.forward(
                    gene_pos,
                    expression.sum(1),
                    expr,
                    full_depth=total_count * true_rescale,  # TODO: add rescaled count
                    timepoint=timepoint,
                )
                l, tot = self._compute_loss(
                    output, expression, None, clss, do_ecs, do_adv_cls, do_mvc
                )
                cell_embs.append(output["cell_emb"].clone())
                total_loss += tot
                losses.update(
                    {"denoise_" + str(int(i * 100)) + "%_" + k: v for k, v in l.items()}
                )
                # make sure that the cell embedding stay the same even if the expression is decreased

        # TASK 4. contrastive cell embedding
        if do_cce:
            # We would want the edge and cell embedding to stay the same
            # Here the idea is that by running the encoder twice, we will have
            # the embeddings for different dropout masks. They will act as a regularizer
            # like presented in https://arxiv.org/pdf/2104.08821.pdf

            # Gather embeddings from all devices if distributed training
            # if dist.is_initialized() and self.training:
            #     cls1_list = [
            #         torch.zeros_like(cell_emb) for _ in range(dist.get_world_size())
            #     ]
            #     cls2_list = [
            #         torch.zeros_like(cell_emb2) for _ in range(dist.get_world_size())
            #     ]
            #     dist.all_gather(tensor_list=cls1_list, tensor=cell_emb.contiguous())
            #     dist.all_gather(tensor_list=cls2_list, tensor=cell_emb2.contiguous())

            #     # NOTE: all_gather results have no gradients, so replace the item
            #     # of the current rank with the original tensor to keep gradients.
            #     # See https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py#L186
            #     cls1_list[dist.get_rank()] = cell_emb
            #     cls2_list[dist.get_rank()] = cell_emb2

            #     cell_emb = torch.cat(cls1_list, dim=0)
            #     cell_emb2 = torch.cat(cls2_list, dim=0)
            # TODO: should detach the second run cls2? Can have a try
            # TODO: to test, we have a label which I don't get.
            # cell_emb (minibatch, nlabels, embsize)
            cell_emb = cell_embs[0]
            loss_cce = 0
            for cell_emb2 in cell_embs[1:]:
                loss_cce += loss.similarity(
                    cell_emb.unsqueeze(1), cell_emb2.unsqueeze(0), cce_sim
                )  # (nlabels, minibatch, minibatch)
            total_loss += loss_cce
            # TASK 3b. contrastive graph embedding
            losses.update({"cce": loss_cce})

        # TASK 6. expression generation
        if default_embs is not None and do_generate:
            out = self._generate(
                default_embs,
                gene_pos,
                full_depth=total_count,
                depth_mult=expression.sum(1),
            )
            l, tloss = self._compute_loss(
                out,
                expression,
                torch.ones_like(expression),
                clss,
                do_ecs,
                do_adv_cls,
                do_mvc,
            )
            losses.update({"gen_" + k: v for k, v in l.items()})
            total_loss += tloss

        # TASK 7. next time point prediction
        if do_next_tp:
            # output = self.forward(gene_pos, expr, mask, depth, timepoint)
            # l, tot = self._compute_loss(
            #    output, expression, mask, clss, do_ecs, do_adv_cls, do_mvc
            # )
            pass
        if total_loss.isnan():
            for k, v in output.items():
                if v.sum().isnan() or v.sum().isinf():
                    print(k, v.mean())
            print(losses)
            import pdb

            pdb.set_trace()

        # TASK 8. KO profile prediction

        # if we have that information

        # TASK 9. PDgrapher-drug-like perturbation prediction (L1000?)
        #
        return total_loss, losses

    def _compute_loss(
        self,
        output,
        expression,
        mask,
        clss,
        do_ecs=False,
        do_adv_cls=False,
        do_mvc=False,
    ):
        total_loss = 0
        losses = {}
        # TASK 1. reconstruct masked expression
        loss_expr = loss.zinb(
            theta=output["disp"],
            pi=output["zero_logits"],
            mu=output["mean"],
            target=expression,
            mask=mask,
        )
        total_loss += loss_expr
        losses.update({"expr": loss_expr})
        # TODO: if target_expression doesn't know a specific gene's expression. add it to mask.
        # THIS is for cases where we have known unknowns. and for things like targeted L1000 seq
        # kinda like padding

        # TASK 2. predict labels
        if len(self.labels) > 0:
            loss_cls = 0
            loss_adv_cls = 0
            for j, labelname in enumerate(self.labels):
                # setting the labels from index to one hot
                loss_cls += self.class_scale * loss.classification(
                    labelname,
                    pred=output["cls_output_" + labelname],
                    cl=clss[:, j],
                    maxsize=self.labels_counts[labelname],
                    cls_hierarchy=self.cls_hierarchy,
                )
                # TASK 2bis. adversarial label prediction
                if do_adv_cls:
                    raise ValueError("me")
                    # TODO: to rewrite & to test
                    for adv_label in self.labels.keys():
                        if adv_label != labelname:
                            advpred = self.cls_decoders[adv_label](
                                output["cell_embs"][:, 2 + j, :]
                            )
                            if len(self.cls_hierarchy) > 0:
                                # computin hierarchical labels and adding them to cl
                                clhier = self.cls_hierarchy.get(labelname, {})
                                if len(clhier) > 0:
                                    addpred = torch.zeros(
                                        (newcl.shape[0], max(clhier.keys()) - maxsize)
                                    )
                                    for k, v in clhier.items():
                                        addpred[:, k - maxsize] = torch.logsumexp(
                                            advpred[:, v], dim=1
                                        )
                                    advpred = torch.cat([advpred, addpred], dim=1)
                            advpred = nn.functional.sigmoid(advpred)
                            loss_adv_cls += self.class_scale * (
                                nn.functional.binary_cross_entropy_with_logits(
                                    advpred, target=newcl, weight=weight
                                )
                            )
            total_loss += loss_cls
            losses.update({"cls": loss_cls})
        if do_adv_cls:
            total_loss -= loss_adv_cls
            losses.update({"adv_cls": loss_adv_cls})
        # TASK 2ter. cell KO effect prediction
        # (just use a novel class, cell state and predict if cell death or not from it)
        # TODO: add large timepoint and set the KO gene to a KO embedding instead of expression embedding
        # TODO: try to require the gene id to still be predictable (with weight tying)
        if do_mvc:
            # TODO: some hidden hyperparams behind this function and maybe other functions
            loss_expr_mvc = loss.zinb(
                theta=output["mvc_disp"],
                pi=output["mvc_zero_logits"],
                mu=output["mvc_mean"],
                target=expression,
                mask=mask,
            )
            total_loss += loss_expr_mvc
            losses.update({"expr_mvc": loss_expr_mvc})
        # TASK 5. elastic cell similarity
        if do_ecs:
            loss_ecs = self.ecs_scale * loss.ecs(
                output["cell_emb"], ecs_threshold=self.ecs_threshold
            )
            total_loss += loss_ecs
            losses.update({"ecs": loss_ecs})
        return losses, total_loss

    def on_validation_start(self):
        for k, v in self.cls_hierarchy.items():
            self.cls_hierarchy[k] = v.to(self.device)

    def on_validation_epoch_start(self):
        self.embs = None

    def validation_step(
        self,
        batch,
        batch_idx,
    ):
        """
        validation_step defines the validation loop. It is independent of forward

        Args:
            batch (list[Tensor]): shape [Tensor(minibatch, seq_len)*2, Tensor(minibatch, n_labels)]
                the n_labels should be in the same orders as the labels provided in the model
                the first Tensor is gene ids, the second is expression of those gene_pos
            batch_idx (Tensor): shape (minibatch)
            superbatch (list[Tensor], optional): shape [(neighbors, seq_len)*minibatch | None]
                gives out additional expression (on the same gene_pos) for the k
                nearest neighbors of the cell at the same minibatch index in "batch"). Defaults to None.
            superbatch_idx (Tensor, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        val_loss, losses = self._full_training(
            batch,
            do_cce=True,
            do_ecs=True,
            do_mvc=False,
            do_denoise=True,
            noise=[0.3],
        )
        expression = batch["x"]
        gene_pos = batch["genes"]
        depth = batch["depth"]
        # depth = torch.min(
        #    torch.tensor(1),
        #    torch.log2(torch.max(depth, torch.tensor(100)) / 100) / 19,
        # ).to(gene_pos.device)
        if self.embs is not None:
            if self.embs.shape[0] < 10000:
                self._predict(gene_pos, expression, depth)
                self.info = torch.cat([self.info, batch["class"]])
        else:
            self._predict(gene_pos, expression, depth)
            self.info = batch["class"]

        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", val_loss, sync_dist=True)
        # self.validation_step_outputs.append(output[''])
        self.log_dict(losses, sync_dist=True)  # Logging to TensorBoard by default
        return val_loss

    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero:
            print("you are not on the main node. cancelling logging step")
            return
        self.log_umap(gtclass=self.info)

    # TODO: compute classification accuracy metrics
    # def on_validation_epoch_end(self):
    #     for labelname, cl in zip(self.labels, clss):
    #         if len(self.cls_hierarchy) > 0:
    #             if len(self.cls_hierarchy[labelname]) > 0:
    #                 if cl in self.cls_hierarchy[labelname].keys():
    #                     # we have to compute the loss by comparing the known
    #                     # class to the children probabilities
    #                     children = self.cls_hierarchy[labelname][cl]
    #                     # make a tensor of the children probabilities
    #                     cl = torch.zeros_like(output["cls_output_" + labelname])
    #                     for child in children:
    #                         cl[:, child] = 1
    #         loss_cls += nn.functional.cross_entropy(
    #             output["cls_output_" + labelname], cl
    #         )
    #     self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        total_loss, losses = self._full_training(
            batch,
            do_cce=True,
            do_ecs=True,
            do_mvc=True,
            do_denoise=True,
            noise=[0.3],
        )
        self.log("test_loss: ", total_loss, sync_dist=True)
        self.log_dict(losses, sync_dist=True)
        return total_loss

    def get_cell_embs(self, gene_pos, expression):
        """
        Args:
            layer_output(:obj:`Tensor`): shape (minibatch, seq_len, embsize)
            weights(:obj:`Tensor`): shape (minibatch, seq_len), optional and only used
                when :attr:`self.cell_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (minibatch, embsize)
        """
        layer_output = self._encoder(
            gene_pos,
            expression,
        )
        embs = self._get_cell_embs(layer_output)
        return embs

    def _get_cell_embs(self, layer_output):
        if self.cell_emb_style == "cls" and self.labels is not None:
            # (minibatch, embsize)
            cell_emb = layer_output[:, : 2 + len(self.labels)]
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        else:
            raise ValueError(f"Unknown cell_emb_style: {self.cell_emb_style}")
        return cell_emb

    def _generate(
        self,
        cell_embs: Tensor,
        gene_pos: Tensor,
        depth_mult: Tensor,
        full_depth: Optional[Tensor] = None,
        tp: Optional[Tensor] = None,
        gen_iters: int = 1,
    ):
        """
        _generate given cell_embeddings, generate an expression profile

        the goal was to iterate multiple times,
        to create a trajectory and reach a certain state
        should call forward multiple times

        Args:
            cell_emb(:obj:`Tensor`): shape (minibatch, embsize)
            src(:obj:`Tensor`): shape (minibatch, seq_len)
            values(:obj:`Tensor`): shape (minibatch, seq_len), optional
            gen_iters(:obj:`int`): number of generation iterations
            labels(:obj:`Tensor`): shape (batch,), optional
        """
        if tp is not None:
            tp = tp / gen_iters
        for i in range(gen_iters):
            transformer_output = self._encoder(
                cell_embs=cell_embs,
                gene_pos=gene_pos,
                full_depth=full_depth,
                timepoint=tp * (i + 1) if tp is not None else None,
            )  # (minibatch, seq_len, embsize)
            cell_embs = self._get_cell_embs(transformer_output)
        output = self._decoder(transformer_output, depth_mult=depth_mult)
        return output  # (minibatch, seq_len)

    def on_predict_epoch_start(self):
        self.embs = None
        if type(self.transformer) is FlashTransformerEncoder:
            for encoder_layers in self.transformer.blocks:
                encoder_layers.set_seq_parallel(False)

    def predict_step(self, batch, batch_idx):
        """
        embed given gene expression, encode the gene embedding and cell embedding.

        Args:
            gene_pos (Tensor): _description_
            expression (Tensor): _description_

        Returns:
            Tensor: _description_
        """
        expression = batch["x"]
        gene_pos = batch["genes"]
        depth = batch["depth"]
        # depth = torch.min(
        #    torch.tensor(1),
        #    torch.log2(torch.max(total_count, torch.tensor(100)) / 100) / 19,
        # ).to(gene_pos.device)
        return self._predict(gene_pos, expression, depth, keep_output=True)

    def _predict(self, gene_pos, expression, depth, keep_output=True):
        if not self.trainer.is_global_zero:
            print("you are not on the main node. cancelling predict step")
            return
        output = self.forward(gene_pos, expression.sum(1), expression, full_depth=depth)
        cell_embs = output["cell_embs"]
        output = self._generate(
            cell_embs, gene_pos, depth_mult=expression.sum(1), full_depth=depth
        )
        ind = [self.labels.index(i) + 2 for i in self.pred_embedding]
        if not keep_output:
            return {
                "embs": torch.mean(cell_embs[:, ind, :], dim=1),
                "class": torch.stack(
                    [
                        torch.argmax(output["cls_output_" + labelname], dim=1)
                        for labelname in self.labels
                    ]
                ).transpose(0, 1),
                "pos": gene_pos,
                "expr": [output["mean"], output["disp"], output["zero_logits"]],
            }
        if self.embs is None:
            self.embs = torch.mean(cell_embs[:, ind, :], dim=1)
            self.pred = torch.stack(
                [
                    torch.argmax(output["cls_output_" + labelname], dim=1)
                    for labelname in self.labels
                ]
            ).transpose(0, 1)
            self.pos = gene_pos
            self.expr_pred = [output["mean"], output["disp"], output["zero_logits"]]
            return self.expr_pred
        else:
            self.embs = torch.cat([self.embs, torch.mean(cell_embs[:, ind, :], dim=1)])
            self.pred = torch.cat(
                [
                    self.pred,
                    torch.stack(
                        [
                            torch.argmax(output["cls_output_" + labelname], dim=1)
                            for labelname in self.labels
                        ]
                    ).transpose(0, 1),
                ],
            )
            self.pos = torch.cat([self.pos, gene_pos])
            self.expr_pred = [
                torch.cat([self.expr_pred[0], output["mean"]]),
                torch.cat([self.expr_pred[1], output["disp"]]),
                torch.cat([self.expr_pred[2], output["zero_logits"]]),
            ]

    def on_predict_epoch_end(self):
        if not self.trainer.is_global_zero:
            print("you are not on the main node. cancelling logging step")
            return
        self.expr_pred = [
            i.to(device="cpu", dtype=torch.float32) for i in self.expr_pred
        ]
        self.pred = self.pred.to(device="cpu", dtype=torch.float32)
        self.embs = self.embs.to(device="cpu", dtype=torch.float32)
        self.pos = self.pos.to(device="cpu", dtype=torch.int32)
        return self.log_umap()

    def log_umap(self, gtclass=None, name=""):
        colname = ["pred_" + i for i in self.labels]
        obs = np.array(self.pred.to(device="cpu", dtype=torch.int32))
        # label decoders is not cls_decoders. one is a dict to map class codes (ints)
        # to class names the other is the module the predict the class
        if self.label_decoders is not None:
            obs = np.array(
                [
                    [self.label_decoders[self.labels[i]][n] for n in name]
                    for i, name in enumerate(obs.T)
                ]
            ).T

        if gtclass is not None:
            colname += self.labels
            nobs = np.array(gtclass.to(device="cpu", dtype=torch.int32))
            if self.label_decoders is not None:
                nobs = np.array(
                    [
                        [self.label_decoders[self.labels[i]][n] for n in name]
                        for i, name in enumerate(nobs.T)
                    ]
                ).T
            obs = np.hstack([obs, nobs])

        adata = AnnData(
            np.array(self.embs.to(device="cpu", dtype=torch.float32)),
            obs=pd.DataFrame(
                obs,
                columns=colname,
            ),
        )
        for n in self.labels:
            if gtclass is not None:
                tr = utils.translate(adata.obs[n].tolist(), n)
                if tr is not None:
                    adata.obs["conv_" + n] = adata.obs[n].replace(tr)
            tr = utils.translate(adata.obs["pred_" + n].tolist(), n)
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
                        for i in self.labels
                    ],
                    [
                        "conv_pred_" + i
                        if "conv_pred_" + i in adata.obs.columns
                        else "pred_" + i
                        for i in self.labels
                    ],
                )
                for i in pair
            ]
            fig, axs = plt.subplots(
                int(len(color) / 2), 2, figsize=(24, len(color) * 4)
            )
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
                "conv_pred_" + i
                if "conv_pred_" + i in adata.obs.columns
                else "pred_" + i
                for i in self.labels
            ]
            fig, axs = plt.subplots(len(color), 1, figsize=(16, len(color) * 8))
            for i, col in enumerate(color):
                sc.pl.umap(
                    adata,
                    color=col,
                    ax=axs[i],
                    show=False,
                )
        try:
            self.logger.experiment.add_figure(fig)
        except:
            print("couldn't log to tensorboard")
        try:
            self.logger.log_image(key="umaps", images=[fig])
        except:
            print("couldn't log to wandb")
        try:
            mdir = self.logger.save_dir if self.logger.save_dir is not None else "/tmp"
        except:
            mdir = "/tmp"
        adata.write(mdir + "/step_" + str(self.global_step) + "_" + name + ".h5ad")
        return adata

    def _predict_denoised_expression(self, gene_pos, expression, depth):
        """
        Args:
            gene_pos (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            expression (:obj:`Tensor`): token values, shape [batch_size, seq_len]

        Returns:
            dict of output Tensors.
        """
        output = self.forward(gene_pos, expression.sum(1), expression, full_depth=depth)
        return output


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
