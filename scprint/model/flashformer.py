import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional
from functools import partial
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
########
from flashattention import flash_attn_qkvpacked_func


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
        self, qkv, bias=None, causal=None, cu_seqlens=None, max_seqlen=None, mask=None
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
            # softmax_scale=self.softmax_scale,
            # bias=bias,
            # causal=causal,
        )
