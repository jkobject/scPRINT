from typing import Dict, Union
from torch import Tensor, nn
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
        explicit_zero_prob: bool = False,
        n_labels: int = 0,
    ):
        super().__init__()
        self.n_labels = n_labels
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
        )
        self.pred = nn.Linear(d_model, 1)
        self.variance = nn.Linear(d_model, 1)
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        # we don't do it on the labels
        x = x[:, self.n_labels :, :]
        z = self.fc(x)
        pred_value = self.pred(z).squeeze(-1)  # (batch, seq_len)
        var_value = self.variance(z).squeeze(-1)  # (batch, seq_len)
        if not self.explicit_zero_prob:
            return dict(counts=pred_value, probs=var_value)
        zero_logits = self.zero_logit(z).squeeze(-1)  # (batch, seq_len)
        # The sigmoid function is used to map the zero_logits to a probability between 0 and 1.
        zero_probs = torch.sigmoid(zero_logits)
        return dict(counts=pred_value, probs=var_value, zero_probs=zero_probs)
        # TODO: note that the return currently is only for training. Since decoder
        # is not used in the test setting for the integration task, the eval/inference
        # logic is not implemented yet. However, remember to implement it when
        # the decoder is used in any test setting. The inference logic will need
        # to sample from the bernoulli distribution with the zero_probs.


class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        layers: list[int] = [256, 128],
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        layers = [d_model] + layers
        self._decoder = nn.ModuleList()
        for i, l in enumerate(layers[1:]):
            self._decoder.append(nn.Linear(layers[i - 1], l))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(l))
        self.out_layer = nn.Linear(layers[-1], n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class MVCDecoder(nn.Module):
    """
    Decoder for the masked value prediction for cell embeddings.
    """

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        n_labels: int = 1,
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "gene product" or 2. "cell product" 3. "concat query" or 4. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        if arch_style == "cell product":
            self.cell2query = nn.Linear(d_model * n_labels, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_model, bias=False)
        if arch_style in ["gene product", "gene product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1)
            # the pred gene expr values, # (batch, seq_len)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            # zero logits need to based on the cell_emb, because of input exprs
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            # expand cell_emb to (batch, seq_len, embsize)
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)
