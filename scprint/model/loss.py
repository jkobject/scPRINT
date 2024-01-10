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


def multi_masked_nb_loss(
    input: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
):
    """
    Compute the masked negative binomial loss between input and target.
    """
    mask = mask.float()
    masked_log_probs = torch.tensor(
        [
            torch.distributions.NegativeBinomial(
                total_count=target, probs=input
            ).log_prob(target)
            * mask
            for target in targets
        ]
    )
    return -masked_log_probs.sum() / mask.sum()


def masked_zinb_loss(
    pi: torch.Tensor,
    probs: torch.Tensor,
    total_count: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    use_logits: bool = True,
    scale=1e3,
) -> torch.Tensor:
    """
    Compute the masked zero-inflated negative binomial loss between input and target.
    """
    mask = mask.float()
    pi = torch.sigmoid(pi) * mask
    if use_logits:
        nb = torch.distributions.NegativeBinomial(
            total_count=torch.ReLu(total_count), logits=probs
        )
    else:
        nb = torch.distributions.NegativeBinomial(
            total_count=torch.ReLu(total_count), probs=probs
        )
    nb_loss = -nb.log_prob(target) * mask

    zero_inflated_loss = -torch.log(pi + (1 - pi) * torch.exp(nb_loss))

    return (zero_inflated_loss.sum() * scale) / (mask.sum() + scale)


def classifier_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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


def graph_similarity_loss(
    input1: torch.Tensor, input2: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the similarity of 2 generated graphs.
    """
    mask = mask.float()
    loss = F.mse_loss(input1 * mask, input2 * mask, reduction="sum")
    return loss / mask.sum()


def graph_sparsity_loss(input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the sparsity of generated graphs.
    """
    mask = mask.float()
    loss = F.mse_loss(input * mask, torch.zeros_like(input) * mask, reduction="sum")
    return loss / mask.sum()
