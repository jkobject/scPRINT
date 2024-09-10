# Copyright (c) 2023, Tri Dao.

from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup

from .activations import swiglu


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
        bias1: bool = True,
        bias2: bool = True,
        return_residual: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Multi-layer perceptron (MLP) module.

        Args:
            in_features (int): Size of each input sample.
            hidden_features (Optional[int], optional): Size of the hidden layer. Defaults to 4 * in_features.
            out_features (Optional[int], optional): Size of each output sample. Defaults to in_features.
            activation (Callable[[torch.Tensor], torch.Tensor], optional): Activation function. Defaults to F.gelu.
            bias1 (bool, optional): If set to False, the first linear layer will not learn an additive bias. Defaults to True.
            bias2 (bool, optional): If set to False, the second linear layer will not learn an additive bias. Defaults to True.
            return_residual (bool, optional): If set to True, the forward method will return a tuple (output, input). Defaults to False.
            device (Optional[torch.device], optional): The desired device of the parameters. Defaults to None.
            dtype (Optional[torch.dtype], optional): The desired data type of the parameters. Defaults to None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else in_features * 4
        )
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(
            hidden_features, out_features, bias=bias2, **factory_kwargs
        )

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Output tensor, or a tuple (output, input) if return_residual is True.
        """
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)
