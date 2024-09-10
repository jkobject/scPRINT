# Copyright (c) 2024, Tri Dao.

from functools import partial
from typing import Any, Callable, Dict, Optional, Type

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops import StochasticDepth

from .mha import MHA
from .mlp import Mlp

try:
    from .layer_norm import RMSNorm, layer_norm_fn
except ModuleNotFoundError:
    layer_norm_fn = None
    RMSNorm = None


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        mixer_cls: Optional[Callable] = None,
        mlp_cls: Optional[Callable] = None,
        norm_cls: Callable = partial(nn.LayerNorm, eps=1e-6),
        dropout_cls: Type[nn.Dropout] = nn.Dropout,
        prenorm: bool = True,
        resid_dropout1: float = 0.0,
        resid_dropout2: float = 0.0,
        drop_path1: float = 0.0,
        drop_path2: float = 0.0,
        fused_dropout_add_ln: bool = False,
        return_residual: bool = False,
        residual_in_fp32: bool = False,
        sequence_parallel: bool = False,
        mark_shared_params: bool = False,
    ):
        """
        For prenorm=True, this Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Dropout -> Add -> LN -> MHA -> Dropout -> Add -> LN -> MLP, returning both
        the hidden_states (output of the MLP) and the residual.
        This is for performance reasons, as we can fuse the dropout, add and LayerNorm.
        The residual needs to be provided (except for the very first block).

        For prenorm=False, this Block has the same structure as a regular postnorm Transformer
        block: MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add -> LN.

        Args:
            dim (int): the number of features in the input.
            mixer_cls (Optional[Callable], optional): the class to use for the mixer layer. Defaults to None.
            mlp_cls (Optional[Callable], optional): the class to use for the mlp layer. Defaults to None.
            norm_cls (Callable, optional): the class to use for the layer norm. Defaults to partial(nn.LayerNorm, eps=1e-6).
            dropout_cls (Type[nn.Dropout], optional): the class to use for the dropout. Defaults to nn.Dropout.
            prenorm (bool, optional): whether to use pre-norm or post-norm. Defaults to True.
            resid_dropout1 (float, optional): the dropout probability for the first dropout layer. Defaults to 0.0.
            resid_dropout2 (float, optional): the dropout probability for the second dropout layer. Defaults to 0.0.
            drop_path1 (float, optional): the drop path probability for the first drop path layer. Defaults to 0.0.
            drop_path2 (float, optional): the drop path probability for the second drop path layer. Defaults to 0.0.
            fused_dropout_add_ln (bool, optional): whether to fuse the dropout, add and layer norm. Defaults to False.
            return_residual (bool, optional): whether each of the sub-layers (mixer and mlp) will return the residual.
                This is for performance reason: for post-norm architecture, returning the input allows us
                to fuse the backward of nn.Linear with the residual connection.
                Defaults to False.
            residual_in_fp32 (bool, optional): whether to keep the residual in fp32. This is for performance reason:
                for post-norm architecture, keeping the residual in fp32 allows us to fuse the backward of nn.Linear
                with the residual connection. Defaults to False.
            sequence_parallel (bool, optional): whether to use sequence parallelism. Defaults to False.
            mark_shared_params (bool, optional): whether to mark the norm parameters as "shared_params".
                This is useful when we want to sync the norm parameters across workers. Defaults to False.
        """
        super().__init__()
        self.prenorm = prenorm
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.return_residual = return_residual
        self.residual_in_fp32 = residual_in_fp32
        if self.residual_in_fp32:
            assert self.prenorm, "residual_in_fp32 is only compatible with prenorm=True"
        if mixer_cls is None:
            mixer_cls = partial(MHA, num_heads=dim // 64)
        if mlp_cls is None:
            mlp_cls = partial(Mlp, hidden_features=4 * dim)
        self.mixer = mixer_cls(dim)
        self.dropout1 = dropout_cls(resid_dropout1)
        self.drop_path1 = StochasticDepth(drop_path1, mode="row")
        self.norm1 = norm_cls(dim)
        self.mlp = mlp_cls(dim)
        if not isinstance(self.mlp, nn.Identity):
            self.dropout2 = dropout_cls(resid_dropout2)
            self.drop_path2 = StochasticDepth(drop_path2, mode="row")
            self.norm2 = norm_cls(dim)

        if self.fused_dropout_add_ln:
            assert layer_norm_fn is not None, "Triton is not installed"
            assert isinstance(self.norm1, (nn.LayerNorm, RMSNorm)) and isinstance(
                self.dropout1, nn.Dropout
            )

        # TD [2023-01-07]: TODO: During training, if sequence_parallel is False and dropout != 0.0,
        # then the input to each worker in the tensor parallel group will be different.
        # This would produce wrong outputs? Somehow we'd need to sync the RNG state across workers.
        # For now this is not an issue because we always use sequence_parallel=True during training
        # and only use sequence_parallel=False during inference.

        # Mark the norm parameters as "sequence_parallel" so that we run all-reduce on their grads.
        if sequence_parallel:
            for p in self.norm1.parameters():
                p._sequence_parallel = True
            if hasattr(self, "norm2"):
                for p in self.norm2.parameters():
                    p._sequence_parallel = True
        # Mark the norm parameters as "shared_params" so that we sync their values at init.
        if mark_shared_params:
            for p in self.norm1.parameters():
                p._shared_params = True
            if hasattr(self, "norm2"):
                for p in self.norm2.parameters():
                    p._shared_params = True

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def set_seq_parallel(self, val: bool):
        for p in self.norm1.parameters():
            p._sequence_parallel = val
        if hasattr(self, "norm2"):
            for p in self.norm2.parameters():
                p._sequence_parallel = val

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        mixer_subset: Optional[Tensor] = None,
        mixer_kwargs: Optional[Dict[str, Any]] = None,
        return_qkv: bool = False,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states (Tensor): The sequence to be passed to the encoder layer. This is a required argument.
            residual (Optional[Tensor]): This argument is used differently based on the normalization method.
                If postnorm is used, residual should be None. If prenorm is used, hidden_states is updated as Attn/MLP(LN(residual)).
            mixer_subset: This argument is used only for cross-attention.
                If not None, a subset of the input sequence 'x' is taken before applying the query projection.
                This is particularly useful for models like ViT where only the CLS token in the last layer is of interest.
            mixer_kwargs: This argument is used only for cross-attention.
                It is a dictionary of additional arguments to be passed to the mixer.
            return_qkv: If True, the function will return the query, key, and value tensors.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: The output tensor of the encoder layer.
            If return_qkv is True, the function will return a tuple of the output tensor and the query, key, and value tensors.
        """
        if self.prenorm:
            if not self.fused_dropout_add_ln:
                dropped = self.drop_path1(self.dropout1(hidden_states))
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                if self.drop_path1.p == 0 or not self.training:
                    rowscale1 = None
                else:
                    rowscale1 = self.drop_path1(
                        torch.ones(
                            hidden_states.shape[:-1],
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        )
                    )
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm1.weight,
                    self.norm1.bias,
                    residual=residual,
                    eps=self.norm1.eps,
                    dropout_p=self.dropout1.p if self.training else 0.0,
                    rowscale=rowscale1,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    is_rms_norm=isinstance(self.norm1, RMSNorm),
                )
            if mixer_kwargs is None:
                mixer_kwargs = {}
            if mixer_subset is not None:
                mixer_kwargs["mixer_subset"] = mixer_subset
            hidden_states = self.mixer(
                hidden_states, return_qkv=return_qkv, bias=bias, **mixer_kwargs
            )
            if return_qkv:
                qkv = hidden_states[1]
                hidden_states = hidden_states[0]
            if mixer_subset is not None:
                residual = residual[:, mixer_subset]
            if not isinstance(self.mlp, nn.Identity):
                if not self.fused_dropout_add_ln:
                    dropped = self.drop_path2(self.dropout2(hidden_states))
                    residual = (dropped + residual) if residual is not None else dropped
                    hidden_states = self.norm2(
                        residual.to(dtype=self.norm2.weight.dtype)
                    )
                    if self.residual_in_fp32:
                        residual = residual.to(torch.float32)
                else:
                    if self.drop_path2.p == 0 or not self.training:
                        rowscale2 = None
                    else:
                        rowscale2 = self.drop_path2(
                            torch.ones(
                                hidden_states.shape[:-1],
                                device=hidden_states.device,
                                dtype=hidden_states.dtype,
                            )
                        )
                    hidden_states, residual = layer_norm_fn(
                        hidden_states,
                        self.norm2.weight,
                        self.norm2.bias,
                        residual=residual,
                        eps=self.norm2.eps,
                        dropout_p=self.dropout2.p if self.training else 0.0,
                        rowscale=rowscale2,
                        prenorm=True,
                        residual_in_fp32=self.residual_in_fp32,
                        is_rms_norm=isinstance(self.norm2, RMSNorm),
                    )
                hidden_states = self.mlp(hidden_states)
            return (
                (hidden_states, residual)
                if not return_qkv
                else (
                    hidden_states,
                    residual,
                    qkv,
                )
            )
        # if not prenorm (disregard for scPRINT)
        else:
            assert residual is None
            mixer_out = self.mixer(
                hidden_states,
                return_qkv=return_qkv,
                bias=bias,
                **(mixer_kwargs if mixer_kwargs is not None else {})
            )
            if return_qkv:
                qkv = mixer_out[-1]
                mixer_out = mixer_out[:-1]
            if self.return_residual:  # mixer out is actually a pair here
                mixer_out, hidden_states = mixer_out
            if not self.fused_dropout_add_ln:
                hidden_states = self.norm1(
                    (self.drop_path1(self.dropout1(mixer_out)) + hidden_states).to(
                        dtype=self.norm1.weight.dtype
                    )
                )
            else:
                if self.drop_path1.p == 0 or not self.training:
                    rowscale1 = None
                else:
                    rowscale1 = self.drop_path1(
                        torch.ones(
                            mixer_out.shape[:-1],
                            device=mixer_out.device,
                            dtype=mixer_out.dtype,
                        )
                    )
                hidden_states = layer_norm_fn(
                    mixer_out,
                    self.norm1.weight,
                    self.norm1.bias,
                    residual=hidden_states,
                    eps=self.norm1.eps,
                    dropout_p=self.dropout1.p if self.training else 0.0,
                    rowscale=rowscale1,
                    prenorm=False,
                    is_rms_norm=isinstance(self.norm1, RMSNorm),
                )
            if not isinstance(self.mlp, nn.Identity):
                mlp_out = self.mlp(hidden_states)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_states = mlp_out
                if not self.fused_dropout_add_ln:
                    hidden_states = self.norm2(
                        (self.drop_path2(self.dropout2(mlp_out)) + hidden_states).to(
                            dtype=self.norm2.weight.dtype
                        )
                    )
                else:
                    if self.drop_path2.p == 0 or not self.training:
                        rowscale2 = None
                    else:
                        rowscale2 = self.drop_path2(
                            torch.ones(
                                mlp_out.shape[:-1],
                                device=mlp_out.device,
                                dtype=mlp_out.dtype,
                            )
                        )
                    hidden_states = layer_norm_fn(
                        mlp_out,
                        self.norm2.weight,
                        self.norm2.bias,
                        residual=hidden_states,
                        eps=self.norm2.eps,
                        dropout_p=self.dropout2.p if self.training else 0.0,
                        rowscale=rowscale2,
                        prenorm=False,
                        is_rms_norm=isinstance(self.norm2, RMSNorm),
                    )
            return hidden_states if not return_qkv else (hidden_states, qkv)
