# from scprint.base.base_model import BaseModel

from typing import Optional, Dict
from torch import Tensor, optim, nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.distributed as dist
import torch

import lightning as L

from matplotlib import pyplot as plt

import pandas as pd
import scanpy as sc
from anndata import AnnData
import math
from functools import partial
import numpy as np

try:
    from .flash_attn import MHA, Block
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
from .linear_transformer import FastTransformerEncoderWrapper
from .EGT import EGT
from .dsbn import DomainSpecificBatchNorm1d
from . import loss
from ..dataloader import tokenizer
from . import utils


class scPrint(L.LightningModule):
    def __init__(
        self,
        genes: list,
        use_precpt_gene_emb: Optional[np.array] = None,
        gene_pos_enc: Optional[list] = None,
        d_model: int = 512,
        nhead: int = 8,
        d_hid: int = 512,
        edge_dim: int = 12,
        nlayers: int = 6,
        layers_cls: list[int] = [],
        labels: Dict[str, int] = {},
        cls_hierarchy: Dict[str, Dict[int, list[int]]] = {},
        dropout: float = 0.5,
        transformer: str = "fast",
        domain_spec_batchnorm: str = "None",
        expr_emb_style: str = "continuous",  # "binned_pos", "cont_pos"
        n_input_bins: int = 0,
        mvc_decoder: str = "inner product",
        cell_emb_style: str = "cls",
        ecs_threshold: float = 0.3,
        similarity: float = 0.5,
        label_decoders: Optional[Dict[str, Dict[int, str]]] = None,
    ):
        """
        __init__ method for TransformerModel.
        # TODO: add docstrings
        Args:
            genes (list): the genenames with which the model will work
            use_precpt_gene_emb (np.array, optional): The gene embeddings. should be of size len(genes), d_model.
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

        # should be stored somehow
        self.d_model = d_model
        self.edge_dim = edge_dim
        self.nlayers = nlayers
        self.gene_pos_enc = gene_pos_enc
        self.mvc_decoder = mvc_decoder
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.ecs_threshold = ecs_threshold
        # need to store
        self.n_input_bins = n_input_bins
        self.transformer = transformer
        self.labels_counts = labels
        self.labels = list(labels.keys())
        self.cell_emb_style = cell_emb_style
        self.embs = None
        self.label_decoders = label_decoders
        self.pred_embedding = None
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
        # TODO: change the model to encoder() / transformer() / decoder()
        if use_precpt_gene_emb is not None:
            self.gene_encoder = encoders.GeneEncoder(
                len(self.vocab), d_model, weights=use_precpt_gene_emb, freeze=True
            )
            self.use_precpt_gene_emb = True
        else:
            self.gene_encoder = encoders.GeneEncoder(len(self.vocab), d_model)
            self.use_precpt_gene_emb = False

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
        self.label_encoder = encoders.BatchLabelEncoder(len(self.labels) + 3, d_model)
        self.time_encoder = encoders.ContinuousValueEncoder(d_model, dropout)
        self.depth_encoder = encoders.ContinuousValueEncoder(d_model, dropout)
        # self.depth_decoder
        # TODO: add sequencing depth decoding (with weight tying?)

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

        # Faster Model
        # TODO: define them all as a layer type
        # Linear
        if transformer == "linear":
            # linear transformer using the fast transformer package
            self.transformer = FastTransformerEncoderWrapper(
                d_model, nhead, d_hid, nlayers, dropout, "linear"
            )
        # flash
        elif transformer == "flash":
            if MHA is None:
                raise ValueError("flash transformer requires flash package")
            # NOT flash transformer using the special tritton kernel
            # or parallelMHA (add the process group thing and faster)
            mode = partial(
                MHA,
                num_heads=nhead,
                dropout=dropout,
                causal=False,
                use_flash_attn=True,
            )
            # or use parallelBlock where attn & MLP are done in parallel
            encoder_layers = Block(
                dim=d_model,
                mixer_cls=mode,
                prenorm=False,
                # need to set it here for now although it hinders some performances as it returns the residual and I need to see what to do with it
                # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
                # residual_in_fp32=True,
                # sequence_parallel=True for more parallelism
            )
            self.transformer = TransformerEncoder(encoder_layers, nlayers)
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
            encoder_layers = TransformerEncoderLayer(
                d_model, nhead, d_hid, dropout, batch_first=True
            )
            self.transformer = TransformerEncoder(encoder_layers, nlayers)

        # decoders
        # expression
        self.expr_decoder = decoders.ExprDecoder(
            d_model,
            nfirst_labels_to_skip=len(self.labels) + 3,
        )
        # cls decoder
        self.cls_decoders = nn.ModuleDict()
        # should be a very simple classifier for most things
        # (maybe scale with the number of classes) should be 1 layer...
        for label, n_cls in labels.items():
            self.cls_decoders[label] = decoders.ClsDecoder(
                d_model, n_cls, layers=layers_cls
            )

        # expression decoder from batch embbedding
        if mvc_decoder is not None:
            self.mvc_decoder = decoders.MVCDecoder(
                d_model,
                arch_style=mvc_decoder,
            )
        else:
            self.mvc_decoder = None

        if similarity is not None:
            self.sim = loss.Similarity(similarity)

        self.apply(
            partial(
                _init_weights,
                n_layer=nlayers,
                # initializer_range=initializer_range,
                # mup_width_scale=getattr(config, "mup_width_scale", 1.0),
            )
        )
        self.save_hyperparameters()

    def on_fit_start(self):
        for k, v in self.cls_hierarchy.items():
            self.cls_hierarchy[k] = v.to(self.device)

    def _encoder(
        self,
        gene_pos: Tensor,
        expression: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        # (minibatch,) unormalized total counts
        depth: Optional[Tensor] = None,
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
            enc += self.expr_encoder(expression, mask)  # (minibatch, seq_len, embsize)
        # else:
        # exp_enc += torch.zeros_like(gene_pos)

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
                torch.Tensor([list(range(len(self.labels) + 3))] * gene_pos.shape[0])
                .int()
                .to(gene_pos.device)
            )
            if cell_embs is None
            else cell_embs
        )  # (minibatch, embsize)

        # populate
        if depth is not None:
            cell_embs[:, 0, :] += self.depth_encoder(depth)
        if timepoint is not None:
            cell_embs[:, 1, :] = self.time_encoder(timepoint)

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

    def _decoder(self, transformer_output, get_gene_emb=False, do_sample=False):
        output = self.expr_decoder(transformer_output)
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
                            :, 3 + i, :
                        ]  # the first elem is the base cell embedding
                    )
                    for i, labelname in enumerate(self.labels)
                }
            )  # (minibatch, n_cls)
        if self.mvc_decoder is not None and False:
            mvc_output = self.mvc_decoder(
                cell_emb,
                self.cur_gene_token_embs,
            )
            output["mvc_mean"] = mvc_output["mean"]  # (minibatch, seq_len)
            output["mvc_disp"] = mvc_output["disp"]
            output["mvc_zero_logits"] = mvc_output["zero_logits"]

        # if self.do_adv:
        # TODO: do DAB
        # output["dab_output"] = self.grad_reverse_discriminator(cell_emb)

        if get_gene_emb:
            output["gene_embedding"] = transformer_output[
                :, len(self.labels) + 3 :, :
            ]  # (minibatch, seq_len, embsize)
        return output

    def forward(
        self,
        gene_pos: Tensor,
        expression: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        # (minibatch,) unormalized total counts
        depth: Optional[Tensor] = None,
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
        transformer_output = self._encoder(gene_pos, expression, mask, depth, timepoint)
        return self._decoder(transformer_output, get_gene_emb, do_sample)

    def configure_optimizers(self, **kwargs):
        # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        optimizer = optim.Adam(self.parameters(), **kwargs)
        return optimizer

    def training_step(
        self,
        batch,
        batch_idx,
        do_denoise=True,
        noise=[0.3],
        do_cce=True,
        do_ecs=True,
        do_mvc=False,
        do_adv_cls=False,
        do_next_tp=False,
        mask_ratio=[0.15, 0.3],
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
            do_denoise,
            noise,
            do_next_tp,
            do_cce,
            do_ecs,
            do_mvc,
            do_adv_cls,
            mask_ratio,
        )
        self.log("train_loss", total_loss, prog_bar=True)
        self.log_dict(losses, prog_bar=True)

        return total_loss

    def _full_training(
        self,
        batch,
        do_denoise=False,
        noise=[],
        do_next_tp=False,
        do_cce=False,
        do_ecs=False,
        do_mvc=False,
        do_adv_cls=False,
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

        expr = torch.log2(1 + ((expression * 10e4) / total_count[:, None])).to(
            gene_pos.device
        )
        depth = torch.min(
            torch.tensor(1),
            torch.log2(torch.max(total_count, torch.tensor(100)) / 100) / 19,
        ).to(gene_pos.device)
        for i in mask_ratio:
            mask = tokenizer.masker(
                length=gene_pos.shape[1],
                batch_size=gene_pos.shape[0],
                mask_ratio=i,
            ).to(gene_pos.device)
            output = self.forward(gene_pos, expr, mask, depth, timepoint)
            l, tot = self._compute_loss(
                output, expression, mask, clss, do_ecs, do_adv_cls, do_mvc
            )

            cell_embs.append(output["cell_emb"])
            if default_embs is None:
                default_embs = output["cell_embs"]
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
                expr = torch.log2(1 + (expr * 10e4) / (total_count[:, None] * (1 - i)))
                output = self.forward(gene_pos, expr, depth=depth, timepoint=timepoint)
                l, tot = self._compute_loss(
                    output, expression, None, clss, do_ecs, do_adv_cls, do_mvc
                )
                cell_embs.append(output["cell_emb"])
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
                cce_sim = self.sim(
                    cell_emb.unsqueeze(1), cell_emb2.unsqueeze(0)
                )  # (nlabels, minibatch, minibatch)
                labels = (
                    torch.arange(cce_sim.size(0))
                    .long()
                    .to(device=cce_sim.device)
                    # .to(cell_embs.device)
                )
                loss_cce += nn.functional.cross_entropy(cce_sim, labels)
            total_loss += loss_cce
            # TASK 3b. contrastive graph embedding
            losses.update({"cce": loss_cce})

        # TASK 6. expression generation
        out = self._generate(default_embs, gene_pos)
        l, tloss = self._compute_loss(
            out,
            expression,
            torch.ones_like(expression),
            clss,
            do_ecs,
            do_adv_cls,
            do_mvc,
        )

        torch.ones_like(expression)
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
            mu=output["mean"] * output["depth"],
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
                loss_cls += 1000 * loss.classification(
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
                                output["cell_embs"][:, 3 + j, :]
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
                            loss_adv_cls += (
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
                mu=output["mvc_mean"] * output["depth"],
                target=expression,
                mask=mask,
            )
            total_loss += loss_expr_mvc
            losses.update({"expr_mvc": loss_expr_mvc})
        # TASK 5. elastic cell similarity
        if do_ecs:
            loss_ecs = loss.ecs(output["cell_emb"], ecs_threshold=self.ecs_threshold)
            total_loss += loss_ecs
            losses.update({"ecs": loss_ecs})
        return losses, total_loss

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
        total_loss, losses = self._full_training(
            batch,
            do_cce=True,
            do_ecs=True,
            do_mvc=False,
            do_denoise=True,
            noise=[0.3],
        )
        expression = batch["x"]
        gene_pos = batch["genes"]
        total_count = batch["depth"]
        expr = torch.log2(1 + ((expression * 10e4) / total_count[:, None])).to(
            gene_pos.device
        )
        depth = torch.min(
            torch.tensor(1),
            torch.log2(torch.max(total_count, torch.tensor(100)) / 100) / 19,
        ).to(gene_pos.device)
        if self.embs is not None:
            if self.embs.shape[0] < 10000:
                self._predict(gene_pos, expr, depth)
                self.info = torch.cat([self.info, batch["class"]])
        else:
            self._predict(gene_pos, expr, depth)
            self.info = batch["class"]
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", total_loss)
        # self.validation_step_outputs.append(output[''])
        self.log_dict(losses)  # Logging to TensorBoard by default
        return total_loss

    def on_validation_epoch_end(self):
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
        self.log("test_loss: ", total_loss)
        self.log_dict(losses)
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
            cell_emb = layer_output[:, : 3 + len(self.labels)]
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        else:
            raise ValueError(f"Unknown cell_emb_style: {self.cell_emb_style}")
        return cell_emb

    def _generate(
        self,
        cell_embs: Tensor,
        gene_pos: Tensor,
        depth: Tensor = None,
        tp: Tensor = None,
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
                depth=depth,
                timepoint=tp * (i + 1) if tp is not None else None,
            )  # (minibatch, seq_len, embsize)
            cell_embs = self._get_cell_embs(transformer_output)
        output = self._decoder(transformer_output)
        return output  # (minibatch, seq_len)

    def on_predict_epoch_start(self):
        self.embs = None

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
        total_count = batch["depth"]
        expression = torch.log2(1 + ((expression * 10e4) / total_count[:, None])).to(
            gene_pos.device
        )
        depth = torch.min(
            torch.tensor(1),
            torch.log2(torch.max(total_count, torch.tensor(100)) / 100) / 19,
        ).to(gene_pos.device)
        self._predict(gene_pos, expression, depth)

    def _predict(self, gene_pos, expression, depth):
        output = self.forward(gene_pos, expression, mask=None, depth=depth)
        if self.pred_embedding is None:
            ind = [1]
        else:
            ind = [1] + [self.labels.index(i) for i in self.pred_embedding]
        if self.embs is None:
            self.embs = torch.mean(output["cell_embs"][:, ind, :], dim=1)
            self.pred = torch.stack(
                [
                    torch.argmax(output["cls_output_" + labelname], dim=1)
                    for labelname in self.labels
                ]
            ).transpose(0, 1)
            self.expr_pred = [output["mean"], output["disp"], output["zero_logits"]]
        elif self.embs.shape[0] > 10000:
            pass
        else:
            self.embs = torch.cat(
                [self.embs, torch.mean(output["cell_embs"][:, ind, :], dim=1)]
            )
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
            self.expr_pred = [
                torch.cat([self.expr_pred[0], output["mean"]]),
                torch.cat([self.expr_pred[1], output["disp"]]),
                torch.cat([self.expr_pred[2], output["zero_logits"]]),
            ]

    def on_predict_epoch_end(self):
        self.expr_pred = [
            i.to(device="cpu", dtype=torch.float32) for i in self.expr_pred
        ]
        self.pred = self.pred.to(device="cpu", dtype=torch.float32)
        self.embs = self.embs.to(device="cpu", dtype=torch.float32)
        return self.log_umap()

    def log_umap(self, gtclass=None):
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
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.tl.leiden(adata)
        adata.obs = adata.obs.astype("category")
        print(adata)
        if gtclass is not None:
            color = [
                i
                for pair in zip(self.labels, ["pred_" + i for i in self.labels])
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
            color = ["pred_" + i for i in self.labels]
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
        return adata

    def _predict_denoised_expression(self, gene_pos, expression, depth):
        """
        Args:
            gene_pos (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            expression (:obj:`Tensor`): token values, shape [batch_size, seq_len]

        Returns:
            dict of output Tensors.
        """
        output = self.forward(gene_pos, expression, depth=depth)
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
        nn.init.normal_(module.weight, std=initializer_range)

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
