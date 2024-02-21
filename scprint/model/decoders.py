from typing import Dict, Union, Callable
from torch import Tensor, nn
from torch.nn import functional as F
import torch
from . import encoders


class GraphSDEExprDecoder(nn.Module):
    def __init__(self, d_model: int, drift: nn.Module, diffusion: nn.Module):
        """
        Initialize the ExprNeuralSDEDecoder module.

        Parameters:
        d_model (int): The dimension of the model.
        drift (nn.Module): The drift component of the SDE.
        diffusion (nn.Module): The diffusion component of the SDE.
        """
        super().__init__()
        self.d_model = d_model
        self.drift = drift
        self.diffusion = diffusion

    def forward(self, x: Tensor, dt: float) -> Tensor:
        drift = self.drift(x)
        diffusion = self.diffusion(x)
        dW = torch.randn_like(x) * torch.sqrt(dt)
        return x + drift * dt + diffusion * dW


class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nfirst_labels_to_skip: int = 0,
        dropout: float = 0.1,
    ):
        super(ExprDecoder, self).__init__()
        self.nfirst_labels_to_skip = nfirst_labels_to_skip
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.finalfc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
        )
        self.depth_encoder = nn.Sequential(
            encoders.ContinuousValueEncoder(d_model, dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.pred_var_zero = nn.Linear(d_model, 3)
        self.depth_fc = nn.Sequential(
            # nn.Linear(d_model, d_model),
            # nn.LeakyReLU(),
            nn.Linear(d_model, 1),
            nn.ReLU(),
        )

    def forward(self, x: Tensor, depth: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        # we don't do it on the labels
        depth = torch.log2(1 + depth)
        depth = self.depth_encoder(depth).unsqueeze(1)
        x = self.fc(x[:, self.nfirst_labels_to_skip :, :])
        x = self.finalfc(x) + depth
        depth_mult = torch.exp(torch.clamp(self.depth_fc(depth.squeeze(1)), max=20))
        pred_value, var_value, zero_logits = self.pred_var_zero(x).split(
            1, dim=-1
        )  # (batch, seq_len)
        # The sigmoid function is used to map the zero_logits to a probability between 0 and 1.
        return dict(
            mean=F.softmax(pred_value.squeeze(-1), dim=-1) * depth_mult,
            disp=torch.exp(torch.clamp(var_value.squeeze(-1), max=20)),
            zero_logits=zero_logits.squeeze(-1),
        )
        # TODO: note that the return currently is only for training. Since decoder
        # is not used in the test setting for the integration task, the eval/inference
        # logic is not implemented yet. However, remember to implement it when
        # the decoder is used in any test setting. The inference logic will need
        # to sample from the bernoulli distribution with the zero_probs.


class MVCDecoder(nn.Module):
    """
    Decoder for the masked value prediction for cell embeddings.
    """

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        dropout: float = 0.1,
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "cell product" 3. "concat query" or 4. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super(MVCDecoder, self).__init__()
        self.depth_encoder = nn.Sequential(
            encoders.ContinuousValueEncoder(d_model, dropout),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
        )
        self.depth_fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
            nn.ReLU(),
        )
        if arch_style == "inner product":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.pred_var_zero = nn.Linear(d_model, d_model * 3, bias=False)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 3)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 3)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.d_model = d_model

    def forward(
        self,
        cell_emb: Tensor,
        gene_embs: Tensor,
        depth: Tensor,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        depth = torch.log2(1 + (depth / 100))
        depth = self.depth_encoder(depth).unsqueeze(1)

        if self.arch_style == "inner product":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2) + depth  # (batch, embsize, 1)
            # the pred gene expr values, # (batch, seq_len)
            pred, var, zero_logits = self.pred_var_zero(query_vecs).split(
                self.d_model, dim=-1
            )
            pred, var, zero_logits = (
                torch.bmm(pred, cell_emb).squeeze(2),
                torch.bmm(var, cell_emb).squeeze(2),
                torch.bmm(zero_logits, cell_emb).squeeze(2),
            )
            # zero logits need to based on the cell_emb, because of input exprs
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            # expand cell_emb to (batch, seq_len, embsize)
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            pred, var, zero_logits = self.fc2(h).split(1, dim=-1)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            pred, var, zero_logits = self.fc2(h).split(1, dim=-1)
        depth_mult = self.depth_fc(depth.squeeze(1))
        return dict(
            mean=F.softmax(pred, dim=-1) * depth_mult,
            disp=torch.exp(var),
            zero_logits=zero_logits,
        )


class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        layers: list[int] = [256, 128],
        activation: Callable = nn.ReLU,
        dropout: float = 0.1,
    ):
        super(ClsDecoder, self).__init__()
        # module list
        layers = [d_model] + layers
        self.decoder = nn.Sequential()
        for i, l in enumerate(layers[1:]):
            self.decoder.append(nn.Linear(layers[i], l))
            self.decoder.append(nn.LayerNorm(l))
            self.decoder.append(activation())
            self.decoder.append(nn.Dropout(dropout))
        self.out_layer = nn.Linear(layers[-1], n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self.decoder:
            x = layer(x)
        return self.out_layer(x)
