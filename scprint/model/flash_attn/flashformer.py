import os
import sys
from functools import partial
from typing import Callable, Optional

import torch
from torch import Tensor, nn
from torch.nn.init import trunc_normal_
from torchvision.ops import StochasticDepth

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from . import MHA, Block, Mlp  # noqa: E402

try:
    from .layer_norm import layer_norm_fn
except ModuleNotFoundError:
    layer_norm_fn = None

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
        residual_in_fp32: bool = True,
        num_heads_kv: Optional[int] = None,
        checkpointing: bool = False,
        fused_dropout_add_ln: bool = False,
        return_residual: bool = False,
        prenorm: bool = True,
        mlp_ratio: float = 4.0,
        fused_mlp: bool = False,
        fused_bias_fc: bool = False,
        sequence_parallel: bool = False,
        drop_path_rate: float = 0.0,
        use_flash_attn: bool = True,
        weight_init: str = "",
    ):
        """
        FlashTransformerEncoder a transformer encoder with flash attention.

        Args:
            d_model (int): The dimension of the input vectors.
            nhead (int): The number of attention heads.
            nlayers (int): The number of layers in the transformer.
            dropout (float, optional): The dropout rate to apply to the output of the positional encoding. Defaults to 0.1.
            residual_in_fp32 (bool, optional): Whether to force the residual to be in fp32 format. Defaults to True.
            num_heads_kv (_type_, optional): The number of heads for key/value. Defaults to None.
            checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
            fused_dropout_add_ln (bool, optional): Whether to fuse dropout, addition and layer normalization operations. Defaults to False.
            return_residual (bool, optional): Whether to return the residual. Defaults to False.
            prenorm (bool, optional): Whether to use pre-normalization. Defaults to True.
            mlp_ratio (float, optional): The ratio for MLP. Defaults to 4.0.
            fused_mlp (bool, optional): Whether to use fused MLP. Defaults to False.
            fused_bias_fc (bool, optional): Whether to fuse bias and fully connected layers. Defaults to False.
            sequence_parallel (bool, optional): Whether to use sequence parallelism. Defaults to False.
            drop_path_rate (float, optional): The drop path rate. Defaults to 0.0.
            weight_init (str, optional): The weight initialization method. Defaults to "".

        Raises:
            ImportError: Raised when Triton is not installed but fused_dropout_add_ln is set to True.
            NotImplementedError: Raised when an unsupported operation is attempted.
        """
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
                use_flash_attn=use_flash_attn,
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

        self.prenorm = prenorm
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

    def forward(
        self,
        hidden_states: Tensor,
        mask: Optional[Tensor] = None,
        return_qkv=[],
        bias: torch.Tensor = None,
        bias_layer=[],
    ) -> Tensor:
        residual = None
        qkvs = []
        if bias is not None and bias.dim() == 2:
            bias = bias.unsqueeze(0).unsqueeze(0)
        for i, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                residual,
                return_qkv=(i in return_qkv),
                bias=bias if i in bias_layer else None,
            )
            if i in return_qkv:
                qkvs.append(hidden_states[-1])
                hidden_states, residual = (
                    hidden_states[:-1] if self.prenorm else hidden_states
                )
            else:
                hidden_states, residual = (
                    hidden_states if self.prenorm else hidden_states
                )
        if self.prenorm:
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
        return hidden_states if len(return_qkv) == 0 else (hidden_states, qkvs)


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
