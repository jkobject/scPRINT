import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple

##TEMP##
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
########

from .flash_attn.flashformer import FlashSelfAttention

# from model.hashformer import CausalSelfAttention


class SparseLinear(nn.Module):
    def __init__(self, insize, outsize):
        super(SparseLinear, self).__init__()
        self.weights = nn.Parameter(torch.randn(insize, outsize))
        self.bias = nn.Parameter(torch.randn(outsize))

    def forward(self, x):
        return torch.sparse.addmm(self.bias, x, self.weights)


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
        self.edge_input = SparseLinear(edge_feat_size, num_heads)
        self.qkv_proj = nn.Linear(feat_size, feat_size * 3)
        self.gate = SparseLinear(edge_feat_size, num_heads)
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
            self.edge_output = SparseLinear(num_heads, edge_feat_size)
            self.mha_dropout_e = nn.Dropout(dropout)
            self.edge_ffn = nn.Sequential(
                nn.LayerNorm(edge_feat_size),
                SparseLinear(edge_feat_size, edge_feat_size),
                activation,
                SparseLinear(edge_feat_size, edge_feat_size),
                nn.Dropout(dropout),
            )

    # TODO: ADD BACK VIRTUAL NODES (FOR CELL EMBEDDINGS)

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
        # this is a debugger line
        import pdb

        pdb.set_trace()
        nfeat_r1 = nfeat
        efeat_r1 = efeat
        nfeat_ln = self.mha_ln_h(nfeat)
        # efeat_ln = self.mha_ln_e(efeat)
        e_bias = self.edge_input(efeat)
        gates = self.gate(efeat)
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
                qkv = rearrange(
                    qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads
                ).to(dtype=torch.float16)

                e_bias = rearrange(e_bias, "b s1 s2 h -> b h s1 s2").to(
                    dtype=torch.float16
                )
                v_attn, attn_hat = self.flash_attn(qkv, bias=e_bias, gates=gates)

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
                # here we decide to actually use the post softmax results
                attn_hat = F.softmax(attn_hat, dim=2)
                attn_tild = attn_hat * gates
                attn_tild = self.attn_dropout(attn_tild)
                # Compute the weighted sum of values (v_h) using the attention scores (attn_tild)
                v_attn = torch.einsum("blmh,bmkh->blkh", attn_tild, v_h)

        # Scale the aggregated values by degree.
        # TODO: put it as a pytorch jit script
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


class OG_EGTLayer(nn.Module):
    @staticmethod
    @torch.jit.script
    def _egt(
        scale_dot: bool,
        scale_degree: bool,
        num_heads: int,
        dot_dim: int,
        clip_logits_min: float,
        clip_logits_max: float,
        attn_dropout: float,
        attn_maskout: float,
        training: bool,
        QKV: torch.Tensor,
        G: torch.Tensor,
        E: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shp = QKV.shape
        Q, K, V = QKV.view(shp[0], shp[1], -1, num_heads).split(dot_dim, dim=2)

        A_hat = torch.einsum("bldh,bmdh->blmh", Q, K)
        if scale_dot:
            A_hat = A_hat * (dot_dim**-0.5)

        H_hat = A_hat.clamp(clip_logits_min, clip_logits_max) + E

        if mask is None:
            if attn_maskout > 0 and training:
                rmask = torch.empty_like(H_hat).bernoulli_(attn_maskout) * -1e9
                gates = torch.sigmoid(G)  # +rmask
                A_tild = F.softmax(H_hat + rmask, dim=2) * gates
            else:
                gates = torch.sigmoid(G)
                A_tild = F.softmax(H_hat, dim=2) * gates
        else:
            if attn_maskout > 0 and training:
                rmask = torch.empty_like(H_hat).bernoulli_(attn_maskout) * -1e9
                gates = torch.sigmoid(G + mask)
                A_tild = F.softmax(H_hat + mask + rmask, dim=2) * gates
            else:
                gates = torch.sigmoid(G + mask)
                A_tild = F.softmax(H_hat + mask, dim=2) * gates

        if attn_dropout > 0:
            A_tild = F.dropout(A_tild, p=attn_dropout, training=training)

        V_att = torch.einsum("blmh,bmkh->blkh", A_tild, V)

        if scale_degree:
            degrees = torch.sum(gates, dim=2, keepdim=True)
            degree_scalers = torch.log(1 + degrees)
            V_att = V_att * degree_scalers

        V_att = V_att.reshape(shp[0], shp[1], num_heads * dot_dim)
        return V_att, H_hat

    @staticmethod
    @torch.jit.script
    def _egt_edge(
        scale_dot: bool,
        num_heads: int,
        dot_dim: int,
        clip_logits_min: float,
        clip_logits_max: float,
        QK: torch.Tensor,
        E: torch.Tensor,
    ) -> torch.Tensor:
        shp = QK.shape
        Q, K = QK.view(shp[0], shp[1], -1, num_heads).split(dot_dim, dim=2)

        A_hat = torch.einsum("bldh,bmdh->blmh", Q, K)
        if scale_dot:
            A_hat = A_hat * (dot_dim**-0.5)
        H_hat = A_hat.clamp(clip_logits_min, clip_logits_max) + E
        return H_hat

    def __init__(
        self,
        node_width,
        edge_width,
        num_heads,
        node_mha_dropout=0,
        edge_mha_dropout=0,
        node_ffn_dropout=0,
        edge_ffn_dropout=0,
        attn_dropout=0,
        attn_maskout=0,
        activation="elu",
        clip_logits_value=[-5, 5],
        node_ffn_multiplier=2.0,
        edge_ffn_multiplier=2.0,
        scale_dot=True,
        scale_degree=False,
        node_update=True,
        edge_update=True,
    ):
        super().__init__()
        self.node_width = node_width
        self.edge_width = edge_width
        self.num_heads = num_heads
        self.node_mha_dropout = node_mha_dropout
        self.edge_mha_dropout = edge_mha_dropout
        self.node_ffn_dropout = node_ffn_dropout
        self.edge_ffn_dropout = edge_ffn_dropout
        self.attn_dropout = attn_dropout
        self.attn_maskout = attn_maskout
        self.activation = activation
        self.clip_logits_value = clip_logits_value
        self.node_ffn_multiplier = node_ffn_multiplier
        self.edge_ffn_multiplier = edge_ffn_multiplier
        self.scale_dot = scale_dot
        self.scale_degree = scale_degree
        self.node_update = node_update
        self.edge_update = edge_update

        assert not (self.node_width % self.num_heads)
        self.dot_dim = self.node_width // self.num_heads

        self.mha_ln_h = nn.LayerNorm(self.node_width)
        self.mha_ln_e = nn.LayerNorm(self.edge_width)
        self.lin_E = nn.Linear(self.edge_width, self.num_heads)
        if self.node_update:
            self.lin_QKV = nn.Linear(self.node_width, self.node_width * 3)
            self.lin_G = nn.Linear(self.edge_width, self.num_heads)
        else:
            self.lin_QKV = nn.Linear(self.node_width, self.node_width * 2)

        self.ffn_fn = getattr(F, self.activation)
        if self.node_update:
            self.lin_O_h = nn.Linear(self.node_width, self.node_width)
            if self.node_mha_dropout > 0:
                self.mha_drp_h = nn.Dropout(self.node_mha_dropout)

            node_inner_dim = round(self.node_width * self.node_ffn_multiplier)
            self.ffn_ln_h = nn.LayerNorm(self.node_width)
            self.lin_W_h_1 = nn.Linear(self.node_width, node_inner_dim)
            self.lin_W_h_2 = nn.Linear(node_inner_dim, self.node_width)
            if self.node_ffn_dropout > 0:
                self.ffn_drp_h = nn.Dropout(self.node_ffn_dropout)

        if self.edge_update:
            self.lin_O_e = nn.Linear(self.num_heads, self.edge_width)
            if self.edge_mha_dropout > 0:
                self.mha_drp_e = nn.Dropout(self.edge_mha_dropout)

            edge_inner_dim = round(self.edge_width * self.edge_ffn_multiplier)
            self.ffn_ln_e = nn.LayerNorm(self.edge_width)
            self.lin_W_e_1 = nn.Linear(self.edge_width, edge_inner_dim)
            self.lin_W_e_2 = nn.Linear(edge_inner_dim, self.edge_width)
            if self.edge_ffn_dropout > 0:
                self.ffn_drp_e = nn.Dropout(self.edge_ffn_dropout)

    def forward(self, g):
        h, e = g.h, g.e
        mask = g.mask

        h_r1 = h
        e_r1 = e

        h_ln = self.mha_ln_h(h)
        e_ln = self.mha_ln_e(e)

        QKV = self.lin_QKV(h_ln)
        E = self.lin_E(e_ln)

        if self.node_update:
            G = self.lin_G(e_ln)
            V_att, H_hat = self._egt(
                self.scale_dot,
                self.scale_degree,
                self.num_heads,
                self.dot_dim,
                self.clip_logits_value[0],
                self.clip_logits_value[1],
                self.attn_dropout,
                self.attn_maskout,
                self.training,
                0 if "num_vns" not in g else g.num_vns,
                QKV,
                G,
                E,
                mask,
            )

            h = self.lin_O_h(V_att)
            if self.node_mha_dropout > 0:
                h = self.mha_drp_h(h)
            h.add_(h_r1)

            h_r2 = h
            h_ln = self.ffn_ln_h(h)
            h = self.lin_W_h_2(self.ffn_fn(self.lin_W_h_1(h_ln)))
            if self.node_ffn_dropout > 0:
                h = self.ffn_drp_h(h)
            h.add_(h_r2)
        else:
            H_hat = self._egt_edge(
                self.scale_dot,
                self.num_heads,
                self.dot_dim,
                self.clip_logits_value[0],
                self.clip_logits_value[1],
                QKV,
                E,
            )

        if self.edge_update:
            e = self.lin_O_e(H_hat)
            if self.edge_mha_dropout > 0:
                e = self.mha_drp_e(e)
            e.add_(e_r1)

            e_r2 = e
            e_ln = self.ffn_ln_e(e)
            e = self.lin_W_e_2(self.ffn_fn(self.lin_W_e_1(e_ln)))
            if self.edge_ffn_dropout > 0:
                e = self.ffn_drp_e(e)
            e.add_(e_r2)

        g = g.copy()
        g.h, g.e = h, e
        return g

    def __repr__(self):
        rep = super().__repr__()
        rep = (
            rep
            + " ("
            + f"num_heads: {self.num_heads},"
            + f"activation: {self.activation},"
            + f"attn_maskout: {self.attn_maskout},"
            + f"attn_dropout: {self.attn_dropout}"
            + ")"
        )
        return rep
