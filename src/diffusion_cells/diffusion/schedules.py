import torch


def linear_beta_schedule(
    T: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: str | torch.device = "cpu",
):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars
