from typing import Dict, Union, Callable
from torch import Tensor, nn
from torch.nn import functional as F
import torch


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
        """
        ExprDecoder Decoder for the gene expression prediction.

        Will output the mean, variance and zero logits, parameters of a zero inflated negative binomial distribution.

        Args:
            d_model (int): The dimension of the model. This is the size of the input feature vector.
            nfirst_labels_to_skip (int, optional): The number of initial labels to skip in the sequence. Defaults to 0.
            dropout (float, optional): The dropout rate applied during training to prevent overfitting. Defaults to 0.1.
        """
        super(ExprDecoder, self).__init__()
        self.nfirst_labels_to_skip = nfirst_labels_to_skip
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            # nn.LayerNorm(d_model),
            nn.LeakyReLU(),
        )
        self.pred_var_zero = nn.Linear(d_model, 3)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        # we don't do it on the labels
        x = self.fc(x[:, self.nfirst_labels_to_skip :, :])
        pred_value, var_value, zero_logits = self.pred_var_zero(x).split(
            1, dim=-1
        )  # (batch, seq_len)
        # The sigmoid function is used to map the zero_logits to a probability between 0 and 1.
        return dict(
            mean=F.softmax(pred_value.squeeze(-1), dim=-1),
            disp=torch.exp(torch.clamp(var_value.squeeze(-1), max=15)),
            zero_logits=zero_logits.squeeze(-1),
        )


class MVCDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        tot_labels: int = 1,
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
    ) -> None:
        """
        MVCDecoder Decoder for the masked value prediction for cell embeddings.

        Will use the gene embeddings with the cell embeddings to predict the mean, variance and zero logits

        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "cell product" 3. "concat query" or 4. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors. Defaults to nn.Sigmoid.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers. Defaults to nn.PReLU.
        """
        super(MVCDecoder, self).__init__()
        if arch_style == "inner product":
            self.gene2query = nn.Linear(d_model, d_model)
            self.norm = nn.LayerNorm(d_model)
            self.query_activation = query_activation()
            self.pred_var_zero = nn.Linear(d_model, d_model * 3, bias=False)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model * (1 + tot_labels), d_model / 2)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(d_model / 2, 3)
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
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        if self.arch_style == "inner product":
            query_vecs = self.query_activation(self.norm(self.gene2query(gene_embs)))
            pred, var, zero_logits = self.pred_var_zero(query_vecs).split(
                self.d_model, dim=-1
            )
            cell_emb = cell_emb.unsqueeze(2)
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
        return dict(
            mvc_mean=F.softmax(pred, dim=-1),
            mvc_disp=torch.exp(torch.clamp(var, max=15)),
            mvc_zero_logits=zero_logits,
        )


class ClsDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_cls: int,
        layers: list[int] = [256, 128],
        activation: Callable = nn.ReLU,
        dropout: float = 0.1,
    ):
        """
        ClsDecoder Decoder for classification task.

        Args:
            d_model: int, dimension of the input.
            n_cls: int, number of classes.
            layers: list[int], list of hidden layers.
            activation: nn.Module, activation function.
            dropout: float, dropout rate.

        Returns:
            Tensor, shape [batch_size, n_cls]
        """
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
        x = self.decoder(x)
        return self.out_layer(x)
