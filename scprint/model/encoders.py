from torch import nn, Tensor
import torch
from typing import Optional
import math
import numpy as np


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
        dropout: float = 0.1,
        freeze: bool = False,
    ):
        super(GeneEncoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx, _freeze=freeze
        )

        if weights is not None:
            # concat a zero vector to the weight
            # this is to make the embedding of the padding token to be zero
            # weights = torch.cat(
            #    [torch.Tensor(weights), torch.zeros(1, embedding_dim)], dim=0
            # )
            self.embedding.weight.data.copy_(torch.Tensor(weights))
        self.enc_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        x = self.dropout(x)
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
        token_to_pos: dict[str, int],  # [token, pos]
        dropout: float = 0.1,
        maxval=10000.0,
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)

        # Create a dictionary to convert token to position

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(maxval) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # we reorder them and map them to gene_id (position)
        arr = []
        for k, v in token_to_pos.items():
            arr.append(pe[v - 1].numpy())
        pe = torch.Tensor(np.array(arr))
        self.register_buffer("pe", pe)

    def forward(self, gene_pos: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return self.dropout(
            torch.index_select(self.pe, 0, gene_pos.view(-1)).view(
                gene_pos.shape + (-1,)
            )
        )


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
        super(DPositionalEncoding, self).__init__()
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
        # TODO: try with a continuous value encoder of size 2 (start, end where they are normalized to 0-1)
        x = x + self.pe1[pos_x]
        x = x + self.pe2[pos_y]
        return self.dropout(x)


class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_value: int = 100_000,
        size: int = 1,
    ):
        super(ContinuousValueEncoder, self).__init__()
        self.max_value = max_value
        self.linear1 = nn.Linear(size, d_model)
        self.activation = nn.ReLU()
        # self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        x = x.unsqueeze(-1)

        # use the mask embedding when x=-1
        # mask = (x == -1).float()
        x = torch.clamp(x, min=0, max=self.max_value)
        x = self.activation(self.norm(self.linear1(x)))
        # x = self.linear2(x)
        x = self.dropout(x)
        if mask is not None:
            x = x.masked_fill_(mask.unsqueeze(-1), 0)
        return x


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
        super(CategoryValueEncoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x
