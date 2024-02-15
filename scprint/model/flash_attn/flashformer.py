import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torchvision.ops import StochasticDepth

from typing import Optional, Callable
from functools import partial
import sys
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
########
from .flashEGT import flash_attn_qkvpacked_func
from . import MHA, Block, Mlp
from .layer_norm import layer_norm_fn

FusedMLP = None


def create_mlp_cls(embed_dim, mlp_ratio, act_layer, fused_mlp):
    inner_dim = int(embed_dim * mlp_ratio)
    if not fused_mlp:
        mlp_cls = partial(Mlp, hidden_features=inner_dim, activation=act_layer())
    else:
        mlp_cls = partial(FusedMLP, hidden_features=inner_dim)
    return mlp_cls


class FlashTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        nlayers: int,
        dropout: float = 0.1,
        residual_in_fp32=True,
        num_heads_kv=None,
        checkpointing=False,
        fused_dropout_add_ln=False,
        return_residual=False,
        prenorm=True,
        mlp_ratio=4.0,
        fused_mlp=False,
        fused_bias_fc=False,
        sequence_parallel=False,
        drop_path_rate=0.0,
        weight_init="",
    ):
        super(FlashTransformerEncoder, self).__init__()

        self.blocks = nn.ModuleList()
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, nlayers)
        ]  # stochastic depth decay rule

        for i in range(nlayers):
            mlp = create_mlp_cls(d_model, mlp_ratio, nn.GELU, fused_mlp)
            attention = partial(
                MHA,
                num_heads=nhead,
                dropout=dropout,
                causal=False,
                use_flash_attn=True,
                num_heads_kv=num_heads_kv,
                checkpointing=checkpointing,
                fused_bias_fc=fused_bias_fc,
                layer_idx=i,
            )
            # or use parallelBlock where attn & MLP are done in parallel
            encoder_layers = Block(
                d_model,
                attention,
                mlp,
                prenorm=prenorm,
                # need to set it here for now although it hinders some performances as it returns the residual and I need to see what to do with it
                # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
                residual_in_fp32=residual_in_fp32,
                sequence_parallel=sequence_parallel,  # for more parallelism
                resid_dropout1=dropout,
                resid_dropout2=dropout,
                drop_path1=dpr[i - 1] if i > 0 else 0.0,
                drop_path2=dpr[i],
                fused_dropout_add_ln=fused_dropout_add_ln,
                return_residual=return_residual,
            )
            self.blocks.append(encoder_layers)

        self.dropout = nn.Dropout(p=dropout)
        self.drop_path = StochasticDepth(p=dpr[-1], mode="row")
        self.norm = torch.nn.LayerNorm(d_model, eps=1e-6)

        self.fused_dropout_add_ln = fused_dropout_add_ln
        if self.fused_dropout_add_ln and layer_norm_fn is None:
            raise ImportError("Triton is not installed")

        if sequence_parallel:
            # This seems to only be important when doing tensor parallelism across GPUs, to increase even more the context length I guess?
            # not really necessary here I think
            raise NotImplementedError("sequence_parallel not implemented yet")

        self.init_weights(weight_init)

    def init_weights(self, mode=""):
        assert mode == ""
        named_apply(_init_weights, self)

    def forward(self, hidden_states: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        residual = None
        for block in self.blocks:
            hidden_states, residual = block(hidden_states, residual)
        if not self.fused_dropout_add_ln:
            residual = self.drop_path(self.dropout(hidden_states)) + residual
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        else:
            if self.drop_path.p == 0 or not self.training:
                rowscale = None
            else:
                rowscale = self.drop_path(
                    torch.ones(
                        hidden_states.shape[:-1],
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                )
            # Set prenorm=False here since we don't need to the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                eps=self.norm.eps,
                dropout_p=self.dropout.p if self.training else 0.0,
                rowscale=rowscale,
                prenorm=False,
            )
        return hidden_states


def _init_weights(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class FlashSelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
            (default: 1/sqrt(d_keys) where d_keys is computed at
            runtime)
        attention_dropout: The dropout rate to apply to the attention
            (default: 0.0)
    """

    def __init__(
        self,
        causal=False,
        softmax_scale=None,
        use_tritton=True,
    ):
        super().__init__()
        # if use_tritton:
        ##TEMP##

        # else:
        #    from flash_attn import flash_attn_qkvpacked_func

        # self.flash_attn_qkvpacked_func = flash_attn_qkvpacked_func

        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward(
        self,
        qkv,
        bias=None,
        gates=None,
        causal=False,
        cu_seqlens=None,
        max_seqlen=None,
        mask=None,
    ):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value.
                If cu_seqlens is None and max_seqlen is None, then qkv has shape (B, S, 3, H, D).
                If cu_seqlens is not None and max_seqlen is not None, then qkv has shape
                (total, 3, H, D), where total is the sum of the sequence lengths in the batch.
            causal: if passed, will override self.causal
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into qkv.
            max_seqlen: int. Maximum sequence length in the batch.
        Returns:
        --------
            out: (total, H, D) if cu_seqlens is not None and max_seqlen is not None,
                else (B, S, H, D).
        """
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        causal = self.causal if causal is None else causal
        return flash_attn_qkvpacked_func(
            qkv,
            bias,
            gates,
            causal,
            self.softmax_scale,
        )
