# from scprint.base.base_model import BaseModel

from typing import Optional, Dict, Union
from torch import Tensor, optim, utils, nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import lightning as L
import torch

import pandas as pd
import math
from functools import partial
import numpy as np

from flash_attn.modules.mha import MHA, ParallelMHA
from flash_attn.modules.mlp import Mlp
from flash_attn.modules.block import Block, ParallelBlock

from . import encoders
from . import decoders
from .linear_transformer import FastTransformerEncoderWrapper
from .hashformer import Hashformer
from .EGT import EGT
from .dsbn import DomainSpecificBatchNorm1d
from .grad_reverse import grad_reverse


class TransformerModel(L.LightningModule):
    def __init__(
        self,
        gene_df: pd.DataFrame,
        d_model: int = 512,
        nhead: int = 8,
        d_hid: int = 512,
        nlayers: int = 6,
        layers_cls: list[int] = [256, 128],
        batches: Optional[dict[str, int]] = {},
        adv_batches: Optional[dict[str, int]] = {},
        dropout: float = 0.5,
        transformer_backend: str = "fast",
        use_precomputed_gene_embbeddings: bool = False,
        do_gene_pos_enc: bool = False,
        do_mvc: bool = False,
        do_dab: bool = False,
        explicit_zero_prob: bool = True,
        domain_spec_batchnorm: Union[bool, str] = False,
        expr_emb_style: str = "continuous",  # "binned_pos", "cont_pos"
        n_input_bins: Optional[int] = 0,
        mvc_decoder_style: str = "inner product",
        ecs_threshold: float = 0.3,
        pre_norm: bool = False,
    ):
        """
        __init__ method for TransformerModel.

        Args:
            d_model (int, optional): The dimension of the model. Defaults to 512.
            nhead (int, optional): The number of heads in the multiheadattention models. Defaults to 8.
            d_hid (int, optional): The dimension of the feedforward network model. Defaults to 512.
            nlayers (int, optional): The number of layers in the transformer model. Defaults to 6.
            nlayers_cls (int, optional): The number of layers in the classifier. Defaults to 3.
            n_cls (int, optional): The number of classes. Defaults to 0.
            dropout (float, optional): The dropout value. Defaults to 0.5.
            do_mvc (bool, optional): Whether to perform MVC. Defaults to False.
            do_dab (bool, optional): Whether to perform DAB. Defaults to False.
            domain_spec_batchnorm (Union[bool, str], optional): Whether to apply domain specific batch normalization. Defaults to False.
            expr_emb_style (str, optional): The style of input embedding (one of "continuous_concat", "binned_pos", "full_pos"). Defaults to "continuous_concat".
            mvc_decoder_style (str, optional): The style of MVC decoder. Defaults to "inner product".
            ecs_threshold (float, optional): The threshold for ECS. Defaults to 0.3.
            pre_norm (bool, optional): Whether to apply pre normalization. Defaults to False.

        Raises:
            ValueError: If the expr_emb_style is not one of "category", "continuous", "none".
        """

        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.do_dab = do_dab
        self.ecs_threshold = ecs_threshold
        self.use_batch_labels = use_batch_labels
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.expr_emb_style = expr_emb_style
        self.norm_scheme = "pre" if pre_norm else "post"
        self.transformer_backend = transformer_backend
        self.use_precomputed_gene_embbeddings = use_precomputed_gene_embbeddings
        self.do_gene_pos_enc = do_gene_pos_enc
        self.mvc_decoder_style = mvc_decoder_style
        self.n_input_bins = n_input_bins
        self.do_mvc = do_mvc
        self.do_dab = do_dab
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.ecs_threshold = ecs_threshold
        self.explicit_zero_prob = explicit_zero_prob
        self.batches = batches
        self.adv_batches = adv_batches
        self.n_batches = len(batches) + len(adv_batches)

        if self.expr_emb_style not in ["category", "continuous", "none"]:
            raise ValueError(
                f"expr_emb_style should be one of category, continuous, scaling, "
                f"got {expr_emb_style}"
            )
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")

        self.gene_df = gene_df
        self.vocab = gene_df["token"].tolist()
        self.ntoken = len(self.vocab)

        # encoder
        # gene encoder
        # TODO: add dropout in the GeneEncoder
        # TODO: move it outside of the model
        if self.use_precomputed_gene_embbeddings:
            gene_emb = gene_df[gene_df.columns.str.startswith("emb_")].values
            self.gene_encoder = encoders.GeneEncoder(
                self.ntoken, d_model, weights=gene_emb, freeze=True
            )
        else:
            self.gene_encoder = encoders.GeneEncoder(self.ntoken, d_model)

        # Value Encoder, NOTE: the scaling style is also handled in _encode method
        if expr_emb_style in ["continuous", "full_pos"]:
            self.value_encoder = encoders.ContinuousValueEncoder(d_model, dropout)
        elif expr_emb_style == "binned_pos":
            assert n_input_bins > 0
            self.value_encoder = encoders.CategoryValueEncoder(n_input_bins, d_model)
        else:
            self.value_encoder = nn.Identity()  # nn.Softmax(dim=1)
            # TODO: consider row-wise normalization or softmax
            # TODO: Correct handle the mask_value when using scaling

        # Positional Encoding
        if self.do_gene_pos_enc:
            gene_pos = gene_df["pos"].values
            self.pos_encoder = encoders.PositionalEncoding(
                d_model, max_len=max(gene_pos)
            )

        # Batch Encoder
        num_batch_labels = len(self.batches) + len(self.adv_batches)
        self.batch_encoder = encoders.BatchLabelEncoder(num_batch_labels, d_model)

        # Model
        # Batch Norm
        if domain_spec_batchnorm is True or domain_spec_batchnorm == "dsbn":
            use_affine = True if domain_spec_batchnorm == "do_affine" else False
            print(f"Use domain specific batchnorm with affine={use_affine}")
            self.dsbn = DomainSpecificBatchNorm1d(
                d_model, num_batch_labels, eps=6.1e-5, affine=use_affine
            )
        elif domain_spec_batchnorm == "batchnorm":
            print("Using simple batchnorm instead of domain specific batchnorm")
            self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)

        # Faster Model
        # TODO: define them all as a layer type
        # Linear
        if transformer_backend == "linear":
            # linear transformer using the fast transformer package
            self.transformer_encoder = FastTransformerEncoderWrapper(
                d_model, nhead, d_hid, nlayers, dropout, "linear"
            )
        # flash
        elif transformer_backend == "flash":
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
                sequence_parallel=True,
                # device?
                batch_first=True,
                norm_scheme=self.norm_scheme,
            )
            # or use parallelBlock where attn & MLP are done in parallel
            encoder_layers = Block(
                dim=d_model,
                mixer_cls=mode,
                # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
                # residual_in_fp32=True,
                # sequence_parallel=True for more parallelism
            )
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # flashsparse
        elif transformer_backend == "flashsparse":
            self.transformer_encoder = Hashformer(
                d_model,
                nlayers,
                2,
                nhead,
            )
        # flash EGT
        elif transformer_backend == "scprint":
            self.transformer_encoder = EGT(
                num_layers=nlayers,
                feat_size=d_model,
                edge_feat_size=edge_d,
                num_heads=nhead,
                num_virtual_nodes=n_virt,
            )
        # regular
        else:
            encoder_layers = TransformerEncoderLayer(
                d_model, nhead, d_hid, dropout, batch_first=True
            )
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # decoders
        # expression
        self.expr_decoder = decoders.ExprDecoder(
            d_model,
            explicit_zero_prob=explicit_zero_prob,
            n_batches=self.n_batches,
        )
        # cls decoder
        for batch, n_cls in self.batches.items():
            self.cls_decoders[batch] = decoders.ClsDecoder(
                d_model, n_cls, layers=layers_cls
            )

        # expression decoder from batch embbedding
        if do_mvc:
            self.mvc_decoder = decoders.MVCDecoder(
                d_model,
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob,
                n_batches=self.n_batches,
            )

        if do_dab:
            self.grad_reverse_discriminator = AdversarialDiscriminator(
                d_model,
                n_cls=num_batch_labels,
                reverse_grad=True,
            )

        self.sim = Similarity(temp=0.5)  # TODO: auto set temp
        self.creterion_cce = nn.CrossEntropyLoss()

        self.apply(
            partial(
                _init_weights,
                n_layer=nlayers,
                # initializer_range=initializer_range,
                # mup_width_scale=getattr(config, "mup_width_scale", 1.0),
            )
        )

    def forward(
        self,
        genes: Tensor,
        expression: Tensor,
        src_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,
        do_class: bool = True,
        do_mvc: bool = True,
        CCE: bool = False,
        ECS: bool = False,
        GEB: bool = False,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
            batch_labels (:obj:`Tensor`): batch labels, shape [batch_size]
            CLS (:obj:`bool`): if True, return the celltype classification objective
                (CLS) output
            CCE (:obj:`bool`): if True, return the contrastive cell embedding objective
                (CCE) output
            MVC (:obj:`bool`): if True, return the masked value prediction for cell
                embedding MVC output
            ECS (:obj:`bool`): if True, return the elastic cell similarity objective
                (ECS) output.
            GEB (:obj:`bool`): if True, return the gene embedding output

        Returns:
            dict of output Tensors.
        """
        transformer_output = self._encode(
            genes, expression, src_key_padding_mask, batch_labels
        )
        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)  # (minibatch, embsize)

        output = {}
        mlm_output = self.decoder(
            transformer_output
            if not self.use_batch_labels
            else torch.cat(
                [
                    transformer_output,
                    batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1),
                ],
                dim=2,
            ),
            # else transformer_output + batch_emb.unsqueeze(1),
        )
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
            output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
        else:
            output["mlm_output"] = mlm_output["pred"]  # (minibatch, seq_len)
        if self.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_output["zero_probs"]

        cell_emb = self._get_cell_emb_from_layer(transformer_output, expression)
        output["cell_emb"] = cell_emb

        if CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)  # (minibatch, n_cls)
        if CCE:
            cell1 = cell_emb
            transformer_output2 = self._encode(
                genes, expression, src_key_padding_mask, batch_labels
            )
            cell2 = self._get_cell_emb_from_layer(transformer_output2)

            # Gather embeddings from all devices if distributed training
            if dist.is_initialized() and self.training:
                cls1_list = [
                    torch.zeros_like(cell1) for _ in range(dist.get_world_size())
                ]
                cls2_list = [
                    torch.zeros_like(cell2) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(tensor_list=cls1_list, tensor=cell1.contiguous())
                dist.all_gather(tensor_list=cls2_list, tensor=cell2.contiguous())

                # NOTE: all_gather results have no gradients, so replace the item
                # of the current rank with the original tensor to keep gradients.
                # See https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py#L186
                cls1_list[dist.get_rank()] = cell1
                cls2_list[dist.get_rank()] = cell2

                cell1 = torch.cat(cls1_list, dim=0)
                cell2 = torch.cat(cls2_list, dim=0)
            # TODO: should detach the second run cls2? Can have a try
            cos_sim = self.sim(
                cell1.unsqueeze(1), cell2.unsqueeze(0)
            )  # (minibatch, batch)
            labels = torch.arange(cos_sim.size(0)).long().to(cell1.device)
            output["loss_cce"] = self.creterion_cce(cos_sim, labels)
        if MVC:
            mvc_output = self.mvc_decoder(
                cell_emb
                if not self.use_batch_labels
                else torch.cat([cell_emb, batch_emb], dim=1),
                # else cell_emb + batch_emb,
                self.cur_gene_token_embs,
            )
            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
                output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]
            else:
                output["mvc_output"] = mvc_output["pred"]  # (minibatch, seq_len)
            if self.explicit_zero_prob:
                output["mvc_zero_probs"] = mvc_output["zero_probs"]
        if ECS:
            # Here using customized cosine similarity instead of F.cosine_similarity
            # to avoid the pytorch issue of similarity larger than 1.0, pytorch # 78064
            # normalize the embedding
            cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
            cos_sim = torch.mm(
                cell_emb_normed, cell_emb_normed.t()
            )  # (minibatch, batch)

            # mask out diagnal elements
            mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
            cos_sim = cos_sim.masked_fill(mask, 0.0)
            # only optimize positive similarities
            cos_sim = F.relu(cos_sim)

            output["loss_ecs"] = torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

        if self.do_dab:
            output["dab_output"] = self.grad_reverse_discriminator(cell_emb)

        if GEB:
            output["gene_embedding"] = transformer_output[
                :, :, :
            ]  # (minibatch, seq_len, embsize)

        return output

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.gene_encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.gene_encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log("val_loss", loss)

    def _encode(
        self,
        genes: Tensor,
        expression: Tensor,
        src_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,  # (batchmini,)
    ) -> Tensor:
        """
        _encode given gene expression, encode the gene embedding and cell embedding.

        Args:
            genes (Tensor): _description_
            expression (Tensor): _description_
            src_key_padding_mask (Tensor): _description_
            batch_labels (Optional[Tensor], optional): _description_. Defaults to None.

        Returns:
            Tensor: _description_
        """
        self._check_batch_labels(batch_labels)
        # TODO add batch label management
        genc = self.gene_encoder(genes)  # (minibatch, seq_len, embsize)
        self.cur_gene_token_embs = genc

        exp_enc = self.value_encoder(expression)  # (minibatch, seq_len, embsize)
        # TODO: how to manage MASK_VALUE?
        if self.expr_emb_style == "scaling":
            exp_enc = exp_enc.unsqueeze(2)
            total_embs = genc * exp_enc
        else:
            total_embs = torch.concat(genc, exp_enc)

        if getattr(self, "dsbn", None) is not None:
            batch_label = int(batch_labels[0].item())
            total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(
                0, 2, 1
            )  # the batch norm always works on dim 1
        elif getattr(self, "bn", None) is not None:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        return output  # (minibatch, seq_len, embsize)

    def _get_cell_emb(
        self, genes, expression, src_key_padding_mask, batch_labels
    ) -> Tensor:
        """
        Args:
            layer_output(:obj:`Tensor`): shape (minibatch, seq_len, embsize)
            weights(:obj:`Tensor`): shape (minibatch, seq_len), optional and only used
                when :attr:`self.cell_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (minibatch, embsize)
        """
        layer_output = self._encode(
            genes,
            expression,
            src_key_padding_mask,
            batch_labels,
        )
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, : len(batch_labels)]  # (minibatch, embsize)
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        else:
            raise ValueError(f"Unknown cell_emb_style: {self.cell_emb_style}")
        return cell_emb

    def _check_batch_labels(self, batch_labels: Tensor) -> None:
        if self.use_batch_labels or self.domain_spec_batchnorm:
            assert batch_labels is not None
        elif batch_labels is not None:
            raise ValueError(
                "batch_labels should only be provided when `self.use_batch_labels`"
                " or `self.domain_spec_batchnorm` is True"
            )

    def _generate(
        self,
        cell_embs: Tensor,
        src: Tensor,
        values: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        gen_iters: int = 1,
        batch_labels: Optional[Tensor] = None,  # (batchmini,)
    ) -> Tensor:
        """
        _generate given cell_embeddings, generate an expression profile

        Args:
            cell_emb(:obj:`Tensor`): shape (minibatch, embsize)
            src(:obj:`Tensor`): shape (minibatch, seq_len)
            values(:obj:`Tensor`): shape (minibatch, seq_len), optional
            src_key_padding_mask(:obj:`Tensor`): shape (minibatch, seq_len), optional
            gen_iters(:obj:`int`): number of generation iterations
            batch_labels(:obj:`Tensor`): shape (batch,), optional
        """
        # TODO: should have a tag indicate the generation mode
        # TODO: if gen_iters > 1, should have a tag indicate the current iteration
        try:
            self._check_batch_labels(batch_labels)
        except:
            import warnings

            warnings.warn(
                "batch_labels is required but not provided, using zeros instead"
            )
            batch_labels = torch.zeros(
                cell_emb.shape[0], dtype=torch.long, device=cell_emb.device
            )

        src = self.gene_encoder(src)  # (minibatch, seq_len, embsize)

        if values is not None:
            values = self.value_encoder(values)  # (minibatch, seq_len, embsize)
            if self.expr_emb_style == "scaling":
                values = values.unsqueeze(2)
                total_embs = src * values
            else:
                total_embs = src + values
        else:
            total_embs = src

        if getattr(self, "dsbn", None) is not None:
            batch_label = int(batch_labels[0].item())
            total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(
                0, 2, 1
            )  # the batch norm always works on dim 1
        elif getattr(self, "bn", None) is not None:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        total_embs[:, 0, :] = cell_emb

        if src_key_padding_mask is None:
            src_key_padding_mask = torch.zeros(
                total_embs.shape[:2], dtype=torch.bool, device=total_embs.device
            )
        transformer_output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )

        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)  # (minibatch, embsize)
        mlm_output = self.decoder(
            transformer_output
            if not self.use_batch_labels
            else torch.cat(
                [
                    transformer_output,
                    batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1),
                ],
                dim=2,
            ),
            # else transformer_output + batch_emb.unsqueeze(1),
        )
        output = mlm_output["pred"]  # (minibatch, seq_len)

        return output  # (minibatch, seq_len)

    def encode_batch(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_size: int,
        batch_labels: Optional[Tensor] = None,
        output_to_cpu: bool = True,
        time_step: Optional[int] = None,
        return_np: bool = False,
    ) -> Tensor:
        """
        Args:
            src (Tensor): shape [N, seq_len]
            values (Tensor): shape [N, seq_len]
            src_key_padding_mask (Tensor): shape [N, seq_len]
            batch_size (int): batch size for encoding
            batch_labels (Tensor): shape [N, n_batch_labels]
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

        for i in trange(0, N, batch_size):
            raw_output = self._encode(
                src[i : i + batch_size].to(device),
                values[i : i + batch_size].to(device),
                src_key_padding_mask[i : i + batch_size].to(device),
                batch_labels[i : i + batch_size].to(device)
                if batch_labels is not None
                else None,
            )
            output = raw_output.detach()
            if output_to_cpu:
                output = output.cpu()
            if return_np:
                output = output.numpy()
            if time_step is not None:
                output = output[:, time_step, :]
            outputs[i : i + batch_size] = output

        return outputs


def generate_square_subsequent_mask(sz: int) -> Tensor:
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


class AdversarialDiscriminator(nn.Module):
    """
    Discriminator for the adversarial training for batch correction.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = False,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


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
