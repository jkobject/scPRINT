import torch.nn.functional as F
import torch

# FROM SCGPT


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


# FROM SCVI


def nb(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, eps=1e-8):
    """
    This negative binomial function was taken from:
    Title: scvi-tools
    Authors: Romain Lopez <romain_lopez@gmail.com>,
             Adam Gayoso <adamgayoso@berkeley.edu>,
             Galen Xing <gx2113@columbia.edu>
    Date: 16th November 2020
    Code version: 0.8.1
    Availability: https://github.com/YosefLab/scvi-tools/blob/8f5a9cc362325abbb7be1e07f9523cfcf7e55ec0/scvi/core/distributions/_negative_binomial.py

    Computes negative binomial loss.
    Parameters
    ----------
    x: torch.Tensor
         Torch Tensor of ground truth data.
    mu: torch.Tensor
         Torch Tensor of means of the negative binomial (has to be positive support).
    theta: torch.Tensor
         Torch Tensor of inverse dispersion parameter (has to be positive support).
    eps: Float
         numerical stability constant.

    Returns
    -------
    If 'mean' is 'True' NB loss value gets returned, otherwise Torch tensor of losses gets returned.
    """
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))

    log_theta_mu_eps = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )

    return res.sum(-1).mean()


def nb_dist(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, eps=1e-8):
    loss = -NegativeBinomial(mu=mu, theta=theta).log_prob(x)
    return loss


def zinb(
    target: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
    eps=1e-8,
    mask=None,
):
    """
    This zero-inflated negative binomial function was taken from:
    Title: scvi-tools
    Authors: Romain Lopez <romain_lopez@gmail.com>,
             Adam Gayoso <adamgayoso@berkeley.edu>,
             Galen Xing <gx2113@columbia.edu>
    Date: 16th November 2020
    Code version: 0.8.1
    Availability: https://github.com/YosefLab/scvi-tools/blob/8f5a9cc362325abbb7be1e07f9523cfcf7e55ec0/scvi/core/distributions/_negative_binomial.py

    Computes zero inflated negative binomial loss.
    Parameters
    ----------
    x: torch.Tensor
         Torch Tensor of ground truth data.
    mu: torch.Tensor
         Torch Tensor of means of the negative binomial (has to be positive support).
    theta: torch.Tensor
         Torch Tensor of inverses dispersion parameter (has to be positive support).
    pi: torch.Tensor
         Torch Tensor of logits of the dropout parameter (real support)
    eps: Float
         numerical stability constant.

    Returns
    -------
    If 'mean' is 'True' ZINB loss value gets returned, otherwise Torch tensor of losses gets returned.
    """
    softplus_pi = F.softplus(-pi)  #  uses log(sigmoid(x)) = -softplus(-x)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((target < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + target * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(target + theta)
        - torch.lgamma(theta)
        - torch.lgamma(target + 1)
    )
    mul_case_non_zero = torch.mul((target > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    # we want to minize the loss but maximize the log likelyhood
    return -res.sum(-1).mean()


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


class Similarity(torch.nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


def ecs(cell_emb, ecs_threshold=0.5):
    # Here using customized cosine similarity instead of F.cosine_similarity
    # to avoid the pytorch issue of similarity larger than 1.0, pytorch # 78064
    # normalize the embedding
    cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
    ecs_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())

    # mask out diagnal elements
    mask = torch.eye(ecs_sim.size(0)).bool().to(ecs_sim.device)
    cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())
    cos_sim = cos_sim.masked_fill(mask, 0.0)
    # only optimize positive similarities
    cos_sim = F.relu(cos_sim)
    return torch.mean(1 - (cos_sim - ecs_threshold) ** 2)


def classification(labelname, pred, cl, maxsize, cls_hierarchy):
    newcl = torch.zeros(
        (cl.shape[0], maxsize), device=cl.device
    )  # batchsize * n_labels
    # if we don't know the label we set the weight to 0 else to 1
    valid_indices = (cl != -1) & (cl < maxsize)
    valid_cl = cl[valid_indices]
    newcl[valid_indices, valid_cl] = 1

    weight = torch.ones_like(newcl, device=cl.device)
    weight[cl == -1, :] = 0
    inv = cl >= maxsize
    # if we have non leaf values, we don't know so we don't compute grad and set weight to 0
    if labelname in cls_hierarchy.keys():
        invw = weight[inv]
        invw[cls_hierarchy[labelname][cl[cl >= maxsize] - maxsize]] = 0
        weight[inv] = invw
    elif inv.any():
        raise ValueError("need to use cls_hierarchy for this usecase")
    # weight[~valid_indices, :] = 0

    if len(cls_hierarchy) > 0:
        if labelname in cls_hierarchy:
            # computing hierarchical labels and adding them to cl
            clhier = cls_hierarchy[labelname]
            npred = pred.unsqueeze(1).expand(-1, clhier.shape[0], -1)
            nnewcl = newcl.unsqueeze(1).expand(-1, clhier.shape[0], -1)

            mask = clhier.unsqueeze(0).expand(npred.shape)
            npred = torch.masked.masked_tensor(npred, mask)
            nnewcl = torch.masked.masked_tensor(nnewcl, mask)
            nnewcl = torch.amax(nnewcl, dim=-1)
            npred = torch.amax(npred, dim=-1)

            addweight = torch.ones(nnewcl.shape).to(pred.device)
            addweight[(cl == -1)] = 0
            newcl = torch.cat([newcl, nnewcl.to_tensor(0)], dim=1)
            pred = torch.cat([pred, npred.to_tensor(0)], dim=1)
            weight = torch.cat([weight, addweight], dim=1)
    myloss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred, target=newcl, weight=weight
    )
    return myloss
