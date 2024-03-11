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
from functools import partial

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
        mvc_decoder: str = "None",
        pred_embedding: list[str] = [],
        cell_emb_style: str = "cls",
        lr: float = 0.001,
        label_decoders: Optional[Dict[str, Dict[int, str]]] = None,
        **flash_attention_kwargs,
    ):
        """
        scPrint transformer for single cell biology and the inference of Gene Regulatory networks

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
            labels (dict, optional): The classes to predict with number of labels for each. Defaults to {}.
            cls_hierarchy (dict, optional): The class hierarchy for classes that have hierarchical labels. Defaults to {}.
            dropout (float, optional): The dropout value. Defaults to 0.5.
            transformer: (flag, optional) the transformer type to use. one of "linear", "flash", "flashsparse", "scprint". Defaults to "flash".
            domain_spec_batchnorm (str, optional): Whether to apply domain specific batch normalization. Defaults to False.
            expr_emb_style (str, optional): The style of input embedding (one of "continuous_concat", "binned_pos", "full_pos"). Defaults to "continuous_concat".
            mvc_decoder (str, optional): The style of MVC decoder one of "None", "inner product", "concat query", "sum query". Defaults to "inner product".
            pred_embedding (list, optional): The list of labels to use for plotting embeddings. Defaults to [].
            cell_emb_style (str, optional): The style of cell embedding. one of "cls", "avg-pool", "w-pool". Defaults to "cls".
            lr (float, optional): The learning rate. Defaults to 0.001.
            label_decoders: (dict, optional) the label decoders to use for plotting the umap during validations. Defaults to None.

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
        self.get_attention_layer = []
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
        # compute tensor for mat_cls_hierarchy
        self.mat_cls_hierarchy = {}
        self.cls_hierarchy = cls_hierarchy
        if "strict_loading" in flash_attention_kwargs:
            flash_attention_kwargs.pop("strict_loading")

        for k, v in cls_hierarchy.items():
            tens = torch.zeros((len(v), labels[k]))
            for k2, v2 in v.items():
                tens[k2 - labels[k], v2] = 1
            self.mat_cls_hierarchy[k] = tens.to(bool)
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
        self.depth_decoder = encoders.ContinuousValueEncoder(d_model, dropout)

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
        # regular or flash
        else:
            if transformer == "flash" and FlashTransformerEncoder is None:
                raise ValueError("flash transformer requires flash package")
                # NOT flash transformer using the special tritton kernel
                # or parallelMHA (add the process group thing and faster)
            self.transformer = FlashTransformerEncoder(
                d_model,
                nhead,
                nlayers,
                dropout=dropout,
                use_flash_attn=(transformer == "flash"),
                **flash_attention_kwargs,
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
        if mvc_decoder != "None":
            self.mvc_decoder = decoders.MVCDecoder(
                d_model,
                arch_style=mvc_decoder,
                dropout=dropout,
            )
        else:
            self.mvc_decoder = None

        self.apply(
            partial(
                utils._init_weights,
                n_layer=nlayers,
            )
        )
        self.save_hyperparameters()

    def _encoder(
        self,
        gene_pos: Tensor,
        expression: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        full_depth: Optional[Tensor] = None,
        timepoint: Optional[Tensor] = None,
        cell_embs: Optional[Tensor] = None,  # (minibatch, n_labels, embsize)
    ):
        """
        _encode given inputs to the model encode into embeddings.

        Args:
            @see self.forward()

        Returns:
            Tensor: the encoded data
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
            cell_embs[:, 1, :] = self.depth_decoder(torch.log2(1 + full_depth))

        enc = torch.cat([cell_embs, enc], dim=1)

        # TODO: seems to be a problem here:
        # if getattr(self, "dsbn", None) is not None and batch_label is not None:
        #     label = int(labels[0].item())
        #     total_embs = self.dsbn(total_embs.permute(0, 2, 1), label).permute(
        #         0, 2, 1
        #     )  # the batch norm always works on dim 1
        # elif getattr(self, "bn", None) is not None:
        #     total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)
        # TODO: get the attention here
        return enc  # (minibatch, seq_len, embsize)

    def _decoder(
        self, transformer_output, depth_mult, get_gene_emb=False, do_sample=False
    ):
        """
        _decoder given the transformer output, decode into the final output.

        Args:
            @see self.forward()

        Returns:
            dict: the output of the model
        """
        output = self.expr_decoder(transformer_output)

        output["mean"] = depth_mult.unsqueeze(1) * output["mean"]
        if do_sample:
            pass
        # bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
        # output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]

        output["cell_embs"] = self.get_cell_embs(transformer_output)
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
        get_attention_layer: list = [],
    ):
        """
        forward also called on self(), a full forward pass on the model

        Args:
            gene_pos (Tensor): A tensor of shape (minibatch, seq_len)
                representing the genes used for each cell in the minibatch.
            expression (Tensor, optional): A tensor of shape (minibatch, seq_len)
                representing the expression levels of genes in the minibatch. Defaults to None.
            mask (Tensor, optional): A tensor of shape (minibatch, seq_len)
                used to mask certain elements in the sequence during the forward pass. Defaults to None.
            full_depth (Tensor, optional): A tensor of shape (minibatch,)
                representing the full depth of each sequence in the minibatch. Defaults to None.
            timepoint (Tensor, optional): A tensor of shape (minibatch,)
                representing the timepoint associated with each sequence in the minibatch. Defaults to None.
            get_gene_emb (bool, optional): A flag indicating whether to return the gene embeddings.
                If True, the gene embeddings are included in the output. Defaults to False.
            do_sample (bool, optional): A flag indicating whether to sample the expression levels.
                If True, the expression levels are sampled during the forward pass. Defaults to False.
            get_attention_layer (list, optional): A list indicating which attention layers to return.
                If not empty, the specified attention layers are included in the output. Defaults to [].

        Returns:
            dict of output Tensors: A dictionary containing the output tensors from the forward pass.
                The keys of the dictionary depend on the input flags (get_gene_emb, do_sample, get_attention_layer).
                at minima, the dictionary contains the following:
                - "mean": the mean expression levels
                - "zero_logits": the logits for zero-inflated expression levels
                - "disp": the dispersion parameter
                - "cell_embs": the cell embeddings per class
                - "cell_emb": the main cell embedding
                - "cls_output": the output of the classifier
        """
        encoding = self._encoder(gene_pos, expression, mask, full_depth, timepoint)
        transformer_output = self.transformer(encoding, return_qkv=get_attention_layer)
        if len(get_attention_layer) > 0:
            transformer_output, qkvs = transformer_output
            return (
                self._decoder(transformer_output, depth_mult, get_gene_emb, do_sample),
                qkvs,
            )
        else:
            return self._decoder(
                transformer_output, depth_mult, get_gene_emb, do_sample
            )

    def configure_optimizers(self):
        """@see pl.LightningModule"""
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
        """@see pl.LightningModule"""
        if type(self.transformer) is FlashTransformerEncoder:
            for encoder_layers in self.transformer.blocks:
                encoder_layers.set_seq_parallel(True)
        for k, v in self.mat_cls_hierarchy.items():
            self.mat_cls_hierarchy[k] = v.to(self.device)

    def training_step(
        self,
        batch: Dict[str, Tensor],
        batch_idx,
    ):
        """
        training_step defines the train loop. It is independent of forward

        @see pl.LightningModule

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

    def _full_training(
        self,
        batch: Dict[str, Tensor],
        do_denoise: bool = False,
        noise: list[float] = [],
        do_next_tp: bool = False,
        do_cce: bool = False,
        cce_sim: float = 0.5,
        do_ecs: bool = False,
        do_mvc: bool = False,
        do_adv_cls: bool = False,
        do_generate: bool = False,
        mask_ratio: list[float] = [0.15],
    ):
        """
        _full_training implement the trainng steps: forward (multiple sometimes), loss

        Args:
            batch (dict[Tensors]): A dictionary containing tensors for the training batch:
                - "x": the expression levels of genes in the minibatch
                - "genes": the genes used for each cell in the minibatch
                - "class": the class to predict for each cell
                - "depth": the full depth of each cell in the minibatch
            do_denoise (bool, optional): A flag to indicate whether to perform denoising. Defaults to False.
            noise (list[float], optional): A list of noise levels to be used in denoising. Defaults to [].
            do_next_tp (bool, optional): A flag to indicate whether to perform next time point prediction. Defaults to False.
            do_cce (bool, optional): A flag to indicate whether to perform cross-categorical entropy. Defaults to False.
            cce_sim (float, optional): The similarity threshold for cross-categorical entropy. Defaults to 0.5.
            do_ecs (bool, optional): A flag to indicate whether to perform elastic cell similarity. Defaults to False.
            do_mvc (bool, optional): A flag to indicate whether to perform multi-view coding. Defaults to False.
            do_adv_cls (bool, optional): A flag to indicate whether to perform adversarial classification. Defaults to False.
            do_generate (bool, optional): A flag to indicate whether to perform data generation. Defaults to False.
            mask_ratio (list, optional): A list of mask ratios to be used in the training. Defaults to [0.15].

        Returns:
            loss, losses: the total loss as float and the individual losses as dict
        """
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
        """
        _compute_loss compute the loss of the model given output from the forward pass

        Args:
            output (dict): A dictionary containing the output of the forward pass.
            expression (Tensor): A tensor containing the expression levels of genes.
            mask (Tensor): A tensor indicating the masked positions in the input data.
            clss (Tensor): A tensor containing the class labels for each cell.
            do_ecs (bool, optional): A flag to indicate whether to perform elastic cell similarity.
                Defaults to False.
            do_adv_cls (bool, optional): A flag to indicate whether to perform adversarial classification.
                Defaults to False.
            do_mvc (bool, optional): A flag to indicate whether to perform masked value prediction for cell embedding.
                Defaults to False.

        Raises:
            ValueError: Raised when an invalid operation or input is encountered.

        Returns:
            tuple: A tuple containing the total loss as a float and the individual losses as a dictionary.
        """
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
                    cls_hierarchy=self.mat_cls_hierarchy,
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
                            if len(self.mat_cls_hierarchy) > 0:
                                # computin hierarchical labels and adding them to cl
                                clhier = self.mat_cls_hierarchy.get(labelname, {})
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

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """@see pl.LightningModule"""
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

    def on_validation_start(self):
        for k, v in self.mat_cls_hierarchy.items():
            self.mat_cls_hierarchy[k] = v.to(self.device)

    def on_validation_epoch_start(self):
        self.embs = None

    def validation_step(
        self,
        batch,
        batch_idx,
    ):
        """
        validation_step defines the validation loop. It is independent of forward
        @see pl.LightningModule

        Args:
            batch (list[Tensor]): @see training_step
        """
        val_loss, losses = self._full_training(
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
        expression = batch["x"]
        gene_pos = batch["genes"]
        depth = batch["depth"]
        if self.embs is not None:
            if self.embs.shape[0] < 10000:
                self._predict(gene_pos, expression, depth)
                self.info = torch.cat([self.info, batch["class"]])
        else:
            self._predict(gene_pos, expression, depth)
            self.info = batch["class"]

        self.log("val_loss", val_loss, sync_dist=True)
        self.log_dict(losses, sync_dist=True)
        return val_loss

    def on_validation_epoch_end(self):
        """@see pl.LightningModule"""
        if not self.trainer.is_global_zero:
            print("you are not on the main node. cancelling logging step")
            return
        self.log_adata(gtclass=self.info)

    # TODO: compute classification accuracy metrics

    def test_step(self, batch, batch_idx):
        """
        @see pl.LightningModule

        Args:
            batch @see training_step
        """
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
        self.log("test_loss: ", total_loss, sync_dist=True)
        self.log_dict(losses, sync_dist=True)
        return total_loss

    def on_predict_epoch_start(self):
        """@see pl.LightningModule"""
        self.embs = None
        self.n_c_batch = 0
        if type(self.transformer) is FlashTransformerEncoder:
            for encoder_layers in self.transformer.blocks:
                encoder_layers.set_seq_parallel(False)

    def predict_step(self, batch, batch_idx):
        """
        embed given gene expression, encode the gene embedding and cell embedding.

        Args:
            batch @see training_step

        Returns:
            Tensor: _description_
        """
        return self._predict(batch["genes"], batch["x"], batch["depth"])

    def _predict(self, gene_pos, expression, depth, keep_output=True):
        """
        @see predict_step will save output of predict in multiple self variables

        - embs: the cell embeddings (means from label specific embeddings given by self.pred_embedding)
        - pred: the predicted cell labels
        - pos: the genes used
        - expr_pred: the expression prediction. [mean, disp, zero_logits]
        - mean_attn: the mean attention across cells for the given layer (in self.get_attention_layer)

        these will be finalized in self.on_predict_epoch_end()

        Args:
            @see training_step
            other important arguments:
            keep_output (bool, optional): whether to keep the output in memory. Defaults to True.
            self.get_attention_layer (list, optional): the layers to get the attention from. Defaults to [].
            self.pred_embedding (list, optional): the labels to predict. Defaults to [].

        """
        if not self.trainer.is_global_zero:
            print("you are not on the main node. cancelling predict step")
            return
        output = self.forward(
            gene_pos,
            expression.sum(1),
            expression,
            full_depth=depth,
            get_attention_layer=self.get_attention_layer,
        )
        if len(self.get_attention_layer) > 0:
            qkv = [i[:, :, :2, :].mean(0) for i in output[1]]
            output = output[0]

        cell_embs = output["cell_embs"]
        # output = self._generate(
        #    cell_embs, gene_pos, depth_mult=expression.sum(1), full_depth=depth
        # )
        if len(self.pred_embedding) == 0:
            self.pred_embedding = self.labels
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
            if len(self.get_attention_layer) > 0:
                self.mean_attn = qkv
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
            if len(self.get_attention_layer) > 0:
                self.mean_attn = [j + qkv[i] for i, j in enumerate(self.mean_attn)]
        self.n_c_batch += 1

    def on_predict_epoch_end(self):
        """@see pl.LightningModule will"""
        if not self.trainer.is_global_zero:
            print("you are not on the main node. cancelling logging step")
            return
        self.expr_pred = [
            i.to(device="cpu", dtype=torch.float32) for i in self.expr_pred
        ]
        self.pred = self.pred.to(device="cpu", dtype=torch.float32)
        self.embs = self.embs.to(device="cpu", dtype=torch.float32)
        self.pos = self.pos.to(device="cpu", dtype=torch.int32)
        if len(self.get_attention_layer) > 0:
            self.mean_attn = [i / self.n_c_batch for i in self.mean_attn]
        return self.log_adata()

    def get_cell_embs(self, layer_output):
        """
        get_cell_embs

        Args:
            layer_output (Tensor): The output tensor from a layer in the model.

        Raises:
            ValueError: Raised when an unknown cell embedding style is encountered.

        Returns:
            Tensor: The cell embeddings tensor.
        """
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
            cell_emb(:obj:`Tensor`): A tensor representing cell embeddings. It has a shape of (minibatch, embsize).
            src(:obj:`Tensor`): A tensor representing the source data. It has a shape of (minibatch, seq_len).
            values(:obj:`Tensor`): An optional tensor representing the values. It has a shape of (minibatch, seq_len).
            gen_iters(:obj:`int`): An integer representing the number of generation iterations.
            labels(:obj:`Tensor`): An optional tensor representing the labels. It has a shape of (batch,).
        """
        if tp is not None:
            tp = tp / gen_iters
        for i in range(gen_iters):
            encoding = self._encoder(
                cell_embs=cell_embs,
                gene_pos=gene_pos,
                full_depth=full_depth,
                timepoint=tp * (i + 1) if tp is not None else None,
            )  # (minibatch, seq_len, embsize)
            transformer_output = self.transformer(encoding)
            cell_embs = self.get_cell_embs(transformer_output)
        output = self._decoder(transformer_output, depth_mult=depth_mult)
        return output  # (minibatch, seq_len)

    def log_adata(self, gtclass=None, name=""):
        """
        log_adata will log an adata from predictions.
        It will log to tensorboard and wandb if available

        see @utils.log_adata
        """
        try:
            mdir = self.logger.save_dir if self.logger.save_dir is not None else "/tmp"
        except:
            mdir = "/tmp"
        adata = utils.make_adata(
            self.pred,
            self.embs,
            self.labels,
            self.trainer.global_step,
            self.label_decoders,
            gtclass,
            name,
            mdir,
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
        output = self.forward(gene_pos, expression.sum(1), expression, full_depth=depth)
        return output
