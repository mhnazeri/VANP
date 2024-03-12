import torch
from torch import nn
import torch.nn.functional as F


def barlow_loss(
    z1, z2, lamda: float = 0.0051, corr_neg_one: bool = False, projection_dim: int = 512
) -> torch.Tensor:
    B = z1.shape[0]
    # normalize the representations along the batch dimension
    z1 = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 0.000006)
    z2 = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 0.000006)
    c = z1.T @ z2
    c.div_(B)
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    if corr_neg_one:
        lamda = 1 / projection_dim
        off_diag = off_diagonal(c).add_(1).pow_(2).sum()
    else:
        off_diag = off_diagonal(c).pow_(2).sum()

    loss = on_diag + lamda * off_diag

    return loss, (on_diag, off_diag)


def vicreg_loss(
    z1, z2, sim_coeff: float = 25.0, std_coeff: float = 25.0, cov_coeff: float = 1.0
):
    repr_loss = F.mse_loss(z1, z2)

    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)

    std_z1 = torch.sqrt(z1.var(dim=0) + 0.0001)
    std_z2 = torch.sqrt(z2.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

    cov_x = (z1.T @ z1) / (z1.shape[0] - 1)
    cov_y = (z2.T @ z2) / (z2.shape[0] - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div_(z1.shape[1]) + off_diagonal(
        cov_y
    ).pow_(2).sum().div_(z2.shape[1])

    loss = sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss
    return loss


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_hat, mean, log_var):
        x = (x - x.min()) / (x.max() - x.min())
        reproduction_loss = nn.functional.binary_cross_entropy(
            x_hat, x, reduction="sum"
        )
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD


class EndToEndLoss(nn.Module):
    def __init__(self, loss_type: "str" = "mse", temp_loss_lambda: int = 0.3):
        super().__init__()
        self.loss_type = loss_type
        self.temp_loss_lambda = temp_loss_lambda

    def forward(self, x_gt, x, dt_gt, dt):
        if self.loss_type == "mse":
            action_loss = torch.nn.functional.mse_loss(x, x_gt)  # (B, T, L)

        elif self.loss_type == "mse+":
            add_term = torch.zeros_like(x_gt, device=x_gt.device, requires_grad=False)
            add_term[:, :, 1] = x_gt[:, :, 0] - x[:, :, 1].detach()
            action_loss = (
                torch.nn.functional.mse_loss(x, x_gt, reduction="none")  # (B, T, L)
                + add_term
            ).mean()

        temp_loss = torch.nn.functional.mse_loss(dt, dt_gt)
        loss = ((1 - self.temp_loss_lambda) * action_loss) + (
            self.temp_loss_lambda * temp_loss
        )
        return loss
