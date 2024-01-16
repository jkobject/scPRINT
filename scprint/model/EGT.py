import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

##TEMP##
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
########

from .flash_attn.flashformer import FlashSelfAttention

# from model.hashformer import CausalSelfAttention


class EGT(nn.Module):
    def __init__(
        self,
        num_layers,
        feat_size,
        edge_feat_size,
        num_heads,
        num_virtual_nodes,
        dropout=0.0,
        attn_dropout=0.0,
        activation=nn.ELU(),
        edge_update=True,
    ):
        super(EGT, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                EGTLayer(
                    feat_size,
                    edge_feat_size,
                    num_heads,
                    num_virtual_nodes,
                    dropout,
                    attn_dropout,
                    activation,
                    edge_update,
                )
            )

    def forward(self, nfeat, efeat):
        for layer in self.layers:
            nfeat, efeat = layer(nfeat, efeat)
        return nfeat, efeat


class EGTLayer(nn.Module):
    r"""EGTLayer for Edge-augmented Graph Transformer (EGT), as introduced in
    `Global Self-Attention as a Replacement for Graph Convolution
    Reference `<https://arxiv.org/pdf/2108.03348.pdf>`_
    modified by @jkobject

    Parameters
    ----------
    feat_size : int
        Node feature size.
    edge_feat_size : int
        Edge feature size.
    num_heads : int
        Number of attention heads, by which :attr: `feat_size` is divisible.
    num_virtual_nodes : int
        Number of virtual nodes.
    dropout : float, optional
        Dropout probability. Default: 0.0.
    attn_dropout : float, optional
        Attention dropout probability. Default: 0.0.
    activation : callable activation layer, optional
        Activation function. Default: nn.ELU().
    edge_update : bool, optional
        Whether to update the edge embedding. Default: True.

    Examples
    --------
    >>> import torch as th
    >>> from dgl.nn import EGTLayer

    >>> batch_size = 16
    >>> num_nodes = 100
    >>> feat_size, edge_feat_size = 128, 32
    >>> nfeat = th.rand(batch_size, num_nodes, feat_size)
    >>> efeat = th.rand(batch_size, num_nodes, num_nodes, edge_feat_size)
    >>> net = EGTLayer(
            feat_size=feat_size,
            edge_feat_size=edge_feat_size,
            num_heads=8,
            num_virtual_nodes=4,
        )
    >>> out = net(nfeat, efeat)
    """

    def __init__(
        self,
        feat_size,
        edge_feat_size,
        num_heads,
        num_virtual_nodes,
        inner_size,
        dropout=0.0,
        scale_dot=False,
        attn_dropout=0.0,
        activation=nn.ELU(),
        edge_update=True,
        hashsparse=False,
        nb_hash=16,
        hashes_per_head=True,
        softmax_scale=1,
        num_heads_kv=None,
        dist_embed=False,
        upto_hop=2,
        svd_encodings=0,
        use_flash=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_virtual_nodes = num_virtual_nodes
        self.edge_update = edge_update
        self.hashsparse = hashsparse
        self.use_flash = use_flash
        self.scale_dot = scale_dot
        # TODO: add distance encoding
        # TODO: add svd encoding (put it outside the model, in scprint)
        if dist_embed:
            self.dist_embed = nn.Embedding(self.upto_hop + 2, self.edge_width)
        else:
            self.dist_embed = None
        if svd_encodings:
            self.svd_embed = nn.Linear(svd_encodings * 2, self.node_width)
        else:
            self.svd_embed = None
        assert feat_size % num_heads == 0, "feat_size must be divisible by num_heads"
        self.dot_dim = feat_size // num_heads
        self.mha_ln_h = nn.LayerNorm(feat_size)
        self.mha_ln_e = nn.LayerNorm(edge_feat_size)
        self.edge_input = nn.Linear(edge_feat_size, num_heads)
        self.qkv_proj = nn.Linear(feat_size, feat_size * 3)
        self.gate = nn.Linear(edge_feat_size, num_heads)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.node_output = nn.Linear(feat_size, feat_size)
        self.mha_dropout_h = nn.Dropout(dropout)
        if self.hashsparse:
            self.hash_attn = CausalSelfAttention(
                num_heads,
                feat_size,
                nb_hash=nb_hash,
                hashes_per_head=hashes_per_head,
                dropout=attn_dropout,
            )
        elif self.use_flash:
            self.flash_attn = FlashSelfAttention(softmax_scale=softmax_scale)
            # self.flash_attn = MHA(
            #    num_heads=num_heads,
            #    embed_dim=feat_size,
            #    num_heads_kv=num_heads_kv,
            #    dropout=dropout,
            #    # process_group
            #    # causal?
            #    # num_heads_kv?
            #    use_flash_attn=True,
            #    sequence_parallel=True,
            #    # device?
            # )
        self.node_ffn = nn.Sequential(
            nn.LayerNorm(feat_size),
            nn.Linear(feat_size, inner_size),
            activation,
            nn.Linear(inner_size, feat_size),
            nn.Dropout(dropout),
        )

        if self.edge_update:
            self.edge_output = nn.Linear(num_heads, edge_feat_size)
            self.mha_dropout_e = nn.Dropout(dropout)
            self.edge_ffn = nn.Sequential(
                nn.LayerNorm(edge_feat_size),
                nn.Linear(edge_feat_size, edge_feat_size),
                activation,
                nn.Linear(edge_feat_size, edge_feat_size),
                nn.Dropout(dropout),
            )

    # def input_block(self, h, e):
    #    dm0 = g.distance_matrix  # (b,i,j)
    #    dm = dm0.long().clamp(max=self.upto_hop + 1)  # (b,i,j)
    #    featm = g.feature_matrix.long()  # (b,i,j,f)
    #
    #    h = self.nodef_embed(nodef).sum(dim=2)  # (b,i,w,h) -> (b,i,h)
    #
    #    if self.svd_encodings:
    #        h = h + self.svd_embed(g.svd_encodings)
    #
    #    e = self.dist_embed(dm) + self.featm_embed(featm).sum(
    #        dim=3
    #    )  # (b,i,j,f,e) -> (b,i,j,e)
    #
    #    if self.num_virtual_nodes > 0:
    #        g = self.vn_layer(g)
    #    return g

    # def output_block(self, g):
    #    h = g.h
    #    h = self.mlp_layers[0](h)
    #    for layer in self.mlp_layers[1:]:
    #        h = layer(self.mlp_fn(h))
    #    return h

    def forward(self, nfeat, efeat, cu_seqlens=None, max_seqlen=None, mask=None):
        """Forward computation. Note: :attr:`nfeat` and :attr:`efeat` should be
        padded with embedding of virtual nodes if :attr:`num_virtual_nodes` > 0,
        while :attr:`mask` should be padded with `0` values for virtual nodes.
        The padding should be put at the beginning.

        Parameters
        ----------
        nfeat : torch.Tensor
            A 3D input tensor. Shape: (batch_size, N, :attr:`feat_size`), where N
            is the sum of the maximum number of nodes and the number of virtual nodes.
        efeat : torch.Tensor
            Edge embedding used for attention computation and self update.
            Shape: (batch_size, N, N, :attr:`edge_feat_size`).
        mask : torch.Tensor, optional
            The attention mask used for avoiding computation on invalid
            positions, where valid positions are indicated by `0` and
            invalid positions are indicated by `-inf`.
            Shape: (batch_size, N, N). Default: None.

        # Max size of efeat is around (16, 2000, 2000, 16) encoded in float16. if we use a kernel able to work on this precision
        # https://developer.nvidia.com/automatic-mixed-precision
        # https://pytorch.org/docs/stable/amp.html
        # https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
        Returns
        -------
        nfeat : torch.Tensor
            The output node embedding. Shape: (batch_size, N, :attr:`feat_size`).
        efeat : torch.Tensor, optional
            The output edge embedding. Shape: (batch_size, N, N, :attr:`edge_feat_size`).
            It is returned only if :attr:`edge_update` is True.
        """
        nfeat_r1 = nfeat
        efeat_r1 = efeat
        nfeat_ln = self.mha_ln_h(nfeat)
        efeat_ln = self.mha_ln_e(efeat)
        e_bias = self.edge_input(efeat_ln)
        gates = self.gate(efeat_ln)
        gates = (
            torch.sigmoid(gates)
            if mask is None
            else torch.sigmoid(gates + mask.unsqueeze(-1))
        )

        if self.hashsparse:
            v_attn = self.hash_attn(nfeat_ln)
            # TODO to add the bias
            # TODO add the gates
            # TODO add the qkv shapes
        else:
            qkv = self.qkv_proj(nfeat_ln)
            bsz, N, _ = qkv.shape
            if self.use_flash:
                # TODO: add the gate to the qkv (it should be a pointwise multiplication with the V)
                # q, k, v = qkv.view(bsz, N, 3, self.num_heads, -1).split(1, dim=2)
                # v = v.squeeze(2) * gates.unsqueeze(-1)
                # qkv = torch.cat([q, k, v], dim=2)
                qkv = rearrange(
                    qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads
                ).to(dtype=torch.float16)
                e_bias = rearrange(e_bias, "b s1 s2 h -> b h s1 s2").to(
                    dtype=torch.float16
                )
                v_attn, attn_tild = self.flash_attn(qkv, bias=e_bias)
                import pdb

                pdb.set_trace()
            else:
                q_h, k_h, v_h = qkv.view(bsz, N, -1, self.num_heads).split(
                    self.dot_dim, dim=2
                )
                # TODO: add flash sparse attention (issue, will need to change the hash_attn file)
                attn_hat = torch.einsum("bldh,bmdh->blmh", q_h, k_h)
                if self.scale_dot:
                    attn_hat = attn_hat * (self.dot_dim**-0.5)
                attn_hat = attn_hat.clamp(-5, 5) + e_bias
                attn_hat = attn_hat if mask is None else attn_hat + mask.unsqueeze(-1)
                attn_hat = F.softmax(attn_hat, dim=2)
                attn_tild = attn_hat * gates
                attn_tild = self.attn_dropout(attn_tild)
                # Compute the weighted sum of values (v_h) using the attention scores (attn_tild)
                v_attn = torch.einsum("blmh,bmkh->blkh", attn_tild, v_h)

        # Scale the aggregated values by degree.
        degrees = torch.sum(gates, dim=2, keepdim=True)
        degree_scalers = torch.log(1 + degrees)
        degree_scalers[:, : self.num_virtual_nodes] = 1.0
        v_attn = v_attn * degree_scalers

        v_attn = v_attn.reshape(bsz, N, self.num_heads * self.dot_dim)
        nfeat = self.node_output(v_attn)

        nfeat = self.mha_dropout_h(nfeat)
        nfeat.add_(nfeat_r1)
        nfeat_r2 = nfeat
        nfeat = self.node_ffn(nfeat)
        nfeat.add_(nfeat_r2)

        if self.edge_update:
            efeat = self.edge_output(attn_hat)
            efeat = self.mha_dropout_e(efeat)
            efeat.add_(efeat_r1)
            efeat_r2 = efeat
            efeat = self.edge_ffn(efeat)
            efeat.add_(efeat_r2)

            return nfeat, efeat

        return nfeat
