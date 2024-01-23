# from scprint.base.base_model import BaseModel

from typing import Optional, Dict, Union, Mapping
from torch import Tensor, optim, nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import lightning as L

import torch.distributed as dist
import torch

import pandas as pd
import math
from functools import partial
import numpy as np

from .flash_attn import MHA, Block

from . import encoders
from . import decoders
from .linear_transformer import FastTransformerEncoderWrapper
from .hashformer import Hashformer
from .EGT import EGT
from .dsbn import DomainSpecificBatchNorm1d
from . import loss
from ..dataloader import tokenizer


class scPrint(L.LightningModule):
    def __init__(
        self,
        genes: list,
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
        use_precpt_gene_emb: Optional[np.array] = None,
        gene_pos_enc: Optional[list] = None,
        domain_spec_batchnorm: str = "None",
        expr_emb_style: str = "continuous",  # "binned_pos", "cont_pos"
        n_input_bins: int = 0,
        mvc_decoder: str = "inner product",
        cell_emb_style: str = "cls",
        ecs_threshold: float = 0.3,
        similarity: float = 0.5,
    ):
        """
        __init__ method for TransformerModel.
        # TODO: add docstrings
        Args:
            genedf (pd.DataFrame): the gene dataframe with columns: index (token names), pos (optional), emb_0, emb_1, ... emb_{d_model-1}
            d_model (int, optional): The dimension of the model. Defaults to 512.
            nhead (int, optional): The number of heads in the multiheadattention models. Defaults to 8.
            d_hid (int, optional): The dimension of the feedforward network model. Defaults to 512.
            nlayers (int, optional): The number of layers in the transformer model. Defaults to 6.
            nlayers_cls (int, optional): The number of layers in the classifier. Defaults to 3.
            n_cls (int, optional): The number of classes. Defaults to 0.
            dropout (float, optional): The dropout value. Defaults to 0.5.
            do_adv (bool, optional): Whether to perform adversarial discrimination. Defaults to False.
            domain_spec_batchnorm (Union[bool, str], optional): Whether to apply domain specific batch normalization. Defaults to False.
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
        self.labels = labels
        self.cell_emb_style = cell_emb_style
        self.cls_hierarchy = cls_hierarchy
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
            self.expr_encoder = nn.Identity()  # nn.Softmax(dim=1)
            # TODO: consider row-wise normalization or softmax
            # TODO: Correct handle the mask_value when using scaling

        # Positional Encoding
        if self.gene_pos_enc is not None:
            max_len = max(gene_pos_enc)
            token_to_pos = {token: pos for token, pos in enumerate(self.gene_pos_enc)}
            self.pos_encoder = encoders.PositionalEncoding(
                d_model, max_len=max_len, token_to_pos=token_to_pos
            )
            # TODO: finish pos encoding

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
            # NOT flash transformer using the special tritton kernel
            # or parallelMHA (add the process group thing and faster)
            mode = MHA(
                num_heads=nhead,
                embed_dim=d_hid,
                dropout=dropout,
                # process_group
                # causal?
                # num_heads_kv?
                use_flash_attn=True,
                # sequence_parallel=True,
                # device?
            )
            # or use parallelBlock where attn & MLP are done in parallel
            encoder_layers = Block(
                dim=d_model,
                mixer_cls=mode,
                # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
                # residual_in_fp32=True,
                # sequence_parallel=True for more parallelism
            )
            self.transformer = TransformerEncoder(encoder_layers, nlayers)
        # flashsparse
        elif transformer == "flashsparse":
            self.transformer = Hashformer(
                d_model,
                nlayers,
                2,
                nhead,
            )
        # flash EGT
        # TODO: We found that the results can be further improved by freezing the
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
        # TODO: should make a very simple classifier for most things (maybe scale with the number of classes) should be 1 layer...
        for label, n_cls in labels.items():
            self.cls_decoders[label] = decoders.ClsDecoder(
                d_model, n_cls, layers=layers_cls
            )

        # expression decoder from batch embbedding
        if mvc_decoder is not None:
            self.mvc_decoder = decoders.MVCDecoder(
                d_model,
                arch_style=mvc_decoder,
                n_labels=len(self.labels),
            )
        else:
            self.mvc_decoder = None

        if similarity is not None:
            self.sim = Similarity(similarity)

        self.apply(
            partial(
                _init_weights,
                n_layer=nlayers,
                # initializer_range=initializer_range,
                # mup_width_scale=getattr(config, "mup_width_scale", 1.0),
            )
        )
        self.save_hyperparameters()

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
        print("encoding")
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
                torch.Tensor(
                    [list(range(len(self.labels) + 3))] * gene_pos.shape[0]
                ).int()
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
        output = self.expr_decoder(transformer_output)
        if do_sample:
            pass
        # bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
        # output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]

        output["cell_embs"] = self._get_cell_embs(transformer_output)
        print(output["cell_embs"].shape)  # batch * n_labels * d_model
        cell_emb = torch.mean(output["cell_embs"], dim=1)
        output["cell_emb"] = cell_emb  # batch * d_model
        if len(self.labels) > 0:
            output.update(
                {
                    "cls_output_"
                    + labelname: self.cls_decoders[labelname](
                        output["cell_embs"][
                            :, 1 + i, :
                        ]  # the first elem is the base cell embedding
                    )
                    for i, labelname in enumerate(self.labels.keys())
                }
            )  # (minibatch, n_cls)
        if self.mvc_decoder is not None:
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

    def configure_optimizers(self, **kwargs):
        # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        optimizer = optim.Adam(self.parameters(), **kwargs)
        return optimizer

    def training_step(
        self,
        batch,
        batch_idx,
        do_denoise=False,
        noise=[0.3],
        do_next_tp=False,
        do_cce=False,
        do_ecs=False,
        do_adv_cls=False,
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

        Returns:
            _type_: _description_
        """
        # TASK 1 & 2 & 3 (first pass, expression reconstruction, label prediction)
        if type(mask_ratio) is not list:
            mask_ratio = [mask_ratio]

        expression = batch[0]
        gene_pos = batch[1]
        clss = batch[2]
        total_count = batch[-1]
        timepoint = batch[-2]

        total_loss = 0
        cell_embs = []

        expr = torch.log2(1 + (expression * 10e4) / total_count[:, None])
        depth = torch.min(
            torch.tensor(1),
            torch.log2(torch.max(total_count, torch.tensor(100)) / 100) / 19,
        )
        for i in mask_ratio:
            mask = tokenizer.masker(
                length=gene_pos.shape[1],
                batch_size=gene_pos.shape[0],
                mask_ratio=i,
            )
            output = self.forward(gene_pos, expr, mask, depth, timepoint)
            l, tot = self._compute_loss(
                output, expression, mask, clss, do_ecs, do_adv_cls
            )
            cell_embs.append(output["cell_emb"])
            total_loss += tot
            losses.update({"mask_" + str(i * 100) + "%_" + k: v for k, v in l.items()})
        # TASK 3. denoising
        if do_denoise:
            for i in noise:
                # Randomly drop on average N counts to each element of expression using a heavy tail Gaussian distribution
                # here we try to get the scale of the distribution so as to remove the right number of counts from each gene
                # TODO: make sure that the scale is correct
                scale = (2 * total_counts * (1 - i)) / (expression > 0).sum(-1)
                drop = torch.poisson(
                    torch.rand(expression.shape) / scale
                ).int()  # Gaussian distribution
                expr = torch.max(expression - drop, torch.zeros_like(expression))
                # TODO: recompute subtotal counts as because of the .max() to keep it to zeros, we might not have removed as much as we thought
                subtotal_count = (total_count * i).int()
                expr = torch.log2(1 + (expr * 10e4) / subtotal_count[:, None])
                mask = tokenizer.masker(
                    length=gene_pos.shape[1],
                    batch_size=gene_pos.shape[0],
                    mask_ratio=mask_ratio,
                )
                output = self.forward(gene_pos, expr, mask, depth, timepoint)
                l, tot = self._compute_loss(
                    output, expression, mask, clss, do_ecs, do_adv_cls
                )
                cell_embs.append(output["cell_emb"])
                total_loss += tot
                losses.update(
                    {"denoise_" + str(i * 100) + "%_" + k: v for k, v in l.items()}
                )
                # make sure that the cell embedding stay the same even if the expression is decreased

        # TASK 4. contrastive cell embedding
        if do_cce:
            # We would want the edge and cell embedding to stay the same
            # Here the idea is that by running the encoder twice, we will have
            # the embeddings for different dropout masks. They will act as a regularizer
            # like presented in https://arxiv.org/pdf/2104.08821.pdf
            # TODO: do we really have different dropouts at each pass?
            # TODO: make sure that dropout is set to 0 after training (for inference)
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
                    torch.arange(cce_sim.size(0)).long()
                    # .to(cell_embs.device)
                )
                loss_cce += nn.functional.cross_entropy(cce_sim, labels)
            total_loss += loss_cce
            # TASK 3b. contrastive graph embedding
            losses.update({"cce": loss_cce})

        # TASK 6. expression generation
        out = self._generate(cell_embs[0], gene_pos)
        l, total_loss = self._compute_loss(
            out,
            expression,
            mask=torch.ones_like(expression),
            clss=clss,
            do_ecs=do_ecs,
            do_adv_cls=do_adv_cls,
        )
        losses.update({"gen_" + k: v for k, v in l.items()})
        total_loss += loss_expr_gene

        # TASK 7. next time point prediction
        if do_next_tp:
            pass

        # TASK 8. KO profile prediction

        # if we have that information

        # TASK 9. PDgrapher-drug-like perturbation prediction
        #
        self.log("train_loss: ", total_loss)
        self.log_dict(losses)

        return total_loss

    def _compute_loss(
        self, output, expression, mask, clss, do_ecs=False, do_adv_cls=False
    ):
        # TASK 1. reconstruct masked expression
        loss_expr = loss.zinb(
            theta=output["disp"],
            pi=output["zero_logits"],
            mu=output["mean"] * output["depth"],
            target=expression,
            mask=mask,
        )
        total_loss = loss_expr
        losses = {"expr": loss_expr}
        # TODO: if target_expression doesn't know a specific gene's expression. add it to mask.
        # THIS is for cases where we have known unknowns. and for things like targeted L1000 seq
        # kinda like padding

        # TASK 2. predict labels
        if len(self.labels) > 0:
            loss_cls = 0
            loss_adv_cls = 0
            import pdb
            pdb.set_trace()
            for j, (labelname, cl) in enumerate(zip(self.labels.keys(), clss.T)):
                # setting the labels from index to one hot
                maxsize = self.labels[labelname]
                newcl = torch.zeros((cl.shape[0], maxsize)) # batchsize * n_labels
                # if we don't know the label we set the weight to 0 else to 1
                weight = torch.ones_like(newcl)
                for i, c in enumerate(cl):
                    if c != -1:
                        if c < maxsize:
                            newcl[i, c] = 1
                        else:
                            # we have cls hierarchy value
                            # we don't know the values below, thus we set them to have 0 effect on the loss
                            weight[i, self.cls_hierarchy[labelname][c]] = 0
                    else:
                        weight[i, :] = 0
                pred = output["cls_output_" + labelname]
                if len(self.cls_hierarchy) > 0:
                    # computin hierarchical labels and adding them to cl
                    clhier = self.cls_hierarchy.get(labelname, {})
                    if len(clhier) > 0:
                        addcl = torch.zeros(
                            (newcl.shape[0], max(clhier.keys()) - maxsize)
                        )
                        addpred = torch.zeros_like(addcl)
                        addweight = torch.ones_like(addcl)
                        for k, v in clhier.items():
                            addcl[:, k - maxsize] = cl[:, v].any(1)
                            addpred[:, k - maxsize] = torch.logsumexp(pred[:, v], dim=1).unsqueeze(1)
                        newcl = torch.cat([cl, newcl], dim=1)
                        pred = torch.cat([pred, addpred], dim=1)
                        weight = torch.cat([weight, addweight], dim=1)

                loss_cls += nn.functional.binary_cross_entropy_with_logits(pred, newcl, weight=weight)
                # TASK 2bis. adversarial label prediction
                if do_adv_cls:
                    for adv_label in self.labels.keys():
                        if adv_label != labelname:
                            adpred = self.cls_decoders[adv_label](
                                output["cell_embs"][
                                    :, 1 + j, :
                                ]
                            )
                            loss_adv_cls += nn.functional.binary_cross_entropy_with_logits(adpred, target=newcl, weight=weight)
            total_loss += -loss_adv_cls + loss_cls
            losses.update({"cls": loss_cls, "adv_cls": loss_adv_cls})
        # TASK 2ter. cell KO effect prediction
        # (just use a novel class, cell state and predict if cell death or not from it)
        # TODO: add large timepoint and set the KO gene to a KO embedding instead of expression embedding
        # TODO: try to require the gene id to still be predictable (with weight tying)

        if self.do_mvc:
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
            # Here using customized cosine similarity instead of F.cosine_similarity
            # to avoid the pytorch issue of similarity larger than 1.0, pytorch # 78064
            # normalize the embedding
            cell_emb_normed = F.normalize(output["cell_emb"], p=2, dim=2)
            output["ecs_sim"] = torch.mm(cell_emb_normed, cell_emb_normed.t())

            # mask out diagnal elements
            mask = (
                torch.eye(output["ecs_sim"].size(0)).bool().to(output["ecs_sim"].device)
            )
            cos_sim = output["cos_sim"].masked_fill(mask, 0.0)
            # only optimize positive similarities
            cos_sim = F.relu(cos_sim)
            loss_ecs = torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)
            total_loss += loss_ecs
            losses.update({"ecs": loss_ecs})
        return losses, total_loss

    def validation_step(self, batch, batch_idx):
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
        gene_pos = batch[0]
        expression = batch[1]
        # TODO: do masking here.
        clss = batch[2:]
        output = self.forward(gene_pos, expression)
        losses, total_loss = self._compute_loss(output, expression, gene_pos, clss)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss: ", total_loss)
        # self.validation_step_outputs.append(output[''])
        self.log_dict(losses)  # Logging to TensorBoard by default
        return total_loss

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
        gene_pos = batch[0]
        expression = batch[1]
        # TODO: do masking here.
        clss = batch[2:]
        output = self.forward(gene_pos, expression, **kwargs)
        losses, total_loss = self._compute_loss(output, expression, gene_pos, clss)
        self.log("test_loss: ", total_loss)
        self.log_dict(losses)
        return total_loss

    def get_cell_emb(self, gene_pos, expression):
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
        return torch.mean(embs, dim=1)

    def _get_cell_embs(self, layer_output):
        if self.cell_emb_style == "cls" and self.labels is not None:
            # (minibatch, embsize)
            cell_emb = layer_output[:, 2 : 3 + len(self.labels)]
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        else:
            raise ValueError(f"Unknown cell_emb_style: {self.cell_emb_style}")
        return cell_emb

    def get_label_emb(self, label):
        """
        get_label_emb given a label, will output a set of embeddings
        that activate this prediction the most in the classifier using LRP.

        Args:
            gene_pos (_type_): _description_
            expression (_type_): _description_

        Returns:
            Tensor: _description_
        """
        # TODO: finish this function
        pass
        # cell_emb[:, self.labels.index]

    # TODO: to finish. should mostly just call forward multiple times
    # the goal was to iterate multiple times,
    # to create a trajectory and reach a certain state
    # should call forward multiple times
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
        output = self.decoder(transformer_output)
        return output  # (minibatch, seq_len)

    # TODO: to finish
    def encode(
        self,
        src: Tensor,
        values: Tensor,
        minibatch_size: int,
        labels: Optional[Tensor] = None,
        output_to_cpu: bool = True,
        time_step: Optional[int] = None,
        return_np: bool = False,
    ):
        """
        encode_batch runs _encode on a large dataset (minibatch per minibatch)

        Args:
            src (Tensor): shape [N, seq_len]
            values (Tensor): shape [N, seq_len]
            minibatch_size (int): batch size for encoding
            labels (Tensor): shape [N, n_labels]
            output_to_cpu (bool): whether to move the output to cpu
            time_step (int): the time step index in the transformer output to return.
                The time step is along the second dimenstion. If None, return all.
            return_np (bool): whether to return numpy array

        Returns:
            output Tensor of shape [N, seq_len, embsize]
        """
        N = src.size(0)
        device = next(self.parameters()).device

        # initialize the output tensor
        array_func = np.zeros if return_np else torch.zeros
        float32_ = np.float32 if return_np else torch.float32
        shape = (
            (N, self.d_model)
            if time_step is not None
            else (N, src.size(1), self.d_model)
        )
        outputs = array_func(shape, dtype=float32_)

        for i in trange(0, N, minibatch_size):
            raw_output = self._encoder(
                src[i : i + minibatch_size].to(device),
                values[i : i + minibatch_size].to(device),
                labels[i : i + minibatch_size].to(device)
                if labels is not None
                else None,
            )
            output = raw_output.detach()
            if output_to_cpu:
                output = output.cpu()
            if return_np:
                output = output.numpy()
            if time_step is not None:
                output = output[:, time_step, :]
            outputs[i : i + minibatch_size] = output

        return outputs


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


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
