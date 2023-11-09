import torch.nn.functional as F
import torch


def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def masked_mae_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MAE loss between input and target.
    MAE = mean absolute error
    """
    mask = mask.float()
    loss = F.l1_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def masked_nb_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked negative binomial loss between input and target.
    """
    mask = mask.float()
    nb = torch.distributions.NegativeBinomial(total_count=target, probs=input)
    masked_log_probs = nb.log_prob(target) * mask
    return -masked_log_probs.sum() / mask.sum()


def classifier_loss(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Compute the cross entropy loss between prediction and target.
    """
    loss = F.cross_entropy(pred, target)
    return loss


def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()

