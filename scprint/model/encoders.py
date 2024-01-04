from torch import nn, Tensor
import torch
from typing import Optional
import math


class GeneEncoder(nn.Module):
    """
    Encodes gene sequences into a continuous vector space using an embedding layer.
    The output is then normalized using a LayerNorm.

    Note: not used in the current version of scprint.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        weights: Optional[Tensor] = None,
        freeze: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx, freeze=freeze
        )
        if weights is not None:
            self.embedding.weight.data.copy_(weights)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class PositionalEncoding(nn.Module):
    """
    The PositionalEncoding module applies a positional encoding to a sequence of vectors.
    This is necessary for the Transformer model, which does not have any inherent notion of
    position in a sequence. The positional encoding is added to the input embeddings and
    allows the model to attend to positions in the sequence.

    Args:
        d_model (int): The dimension of the input vectors.
        dropout (float, optional): The dropout rate to apply to the output of the positional encoding.
        max_len (int, optional): The maximum length of a sequence that this module can handle.

    Note: not used in the current version of scprint.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int,
        dropout: float = 0.1,
        maxval=10000.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(maxval) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor, pos_x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[pos_x]
        return self.dropout(x)


class DPositionalEncoding(nn.Module):
    """
    The PositionalEncoding module applies a positional encoding to a sequence of vectors.
    This is necessary for the Transformer model, which does not have any inherent notion of
    position in a sequence. The positional encoding is added to the input embeddings and
    allows the model to attend to positions in the sequence.

    Args:
        d_model (int): The dimension of the input vectors.
        dropout (float, optional): The dropout rate to apply to the output of the positional encoding.
        max_len (int, optional): The maximum length of a sequence that this module can handle.

    Note: not used in the current version of scprint.
    """

    def __init__(
        self,
        d_model: int,
        max_len_x: int,
        max_len_y: int,
        maxvalue_x=10000.0,
        maxvalue_y=10000.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position2 = torch.arange(max_len_y).unsqueeze(1)
        position1 = torch.arange(max_len_x).unsqueeze(1)

        half_n = d_model // 2

        div_term2 = torch.exp(
            torch.arange(0, half_n, 2) * (-math.log(maxvalue_y) / d_model)
        )
        div_term1 = torch.exp(
            torch.arange(0, half_n, 2) * (-math.log(maxvalue_x) / d_model)
        )
        pe1 = torch.zeros(max_len_x, 1, d_model)
        pe2 = torch.zeros(max_len_y, 1, d_model)
        pe1[:, 0, 0:half_n:2] = torch.sin(position1 * div_term1)
        pe1[:, 0, 1:half_n:2] = torch.cos(position1 * div_term1)
        pe2[:, 0, half_n::2] = torch.sin(position2 * div_term2)
        pe2[:, 0, 1 + half_n :: 2] = torch.cos(position2 * div_term2)
        # https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
        # TODO: seems to do it differently. I hope it still works ok!!
        self.register_buffer("pe1", pe1)
        self.register_buffer("pe2", pe2)

        # PE(x,y,2i) = sin(x/10000^(4i/D))
        # PE(x,y,2i+1) = cos(x/10000^(4i/D))
        # PE(x,y,2j+D/2) = sin(y/10000^(4j/D))
        # PE(x,y,2j+1+D/2) = cos(y/10000^(4j/D))

    def forward(self, x: Tensor, pos_x: Tensor, pos_y: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe1[pos_x]
        x = x + self.pe2[pos_y]
        return self.dropout(x)


class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        # self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, min=0, max=1)
        x = self.activation(self.linear1(x))
        # x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class CategoryValueEncoder(nn.Module):
    """
    Encodes categorical values into a vector using an embedding layer and layer normalization.

    Note: not used in the current version of scprint.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class BatchLabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x


class EGTEncoder:
    def __init__(self, d_model: int, nhead: int, num_layers: int):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.transformer = dgl.nn.EGTLayer(
            feat_size=feat_size,
            edge_feat_size=edge_feat_size,
            num_heads=8,
            num_virtual_nodes=4,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transformer(x)
