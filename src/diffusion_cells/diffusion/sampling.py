import torch
import torch.nn as nn


@torch.no_grad()
def ddpm_inpaint_noised_clamp(
    model: nn.Module,
    x0: torch.Tensor,
    mask: torch.Tensor,
    T: int,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    posterior_variance: torch.Tensor,
    device: torch.device,
):
    model.eval()
    B, C, H, W = x0.shape
    x = torch.randn_like(x0)

    for t in reversed(range(T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        ab = alpha_bars[t]
        eps_known = torch.randn_like(x0)
        x_known_t = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * eps_known

        # clamp known pixels at the correct noise level
        x = x * mask + x_known_t * (1.0 - mask)

        x0_masked = x0 * (1.0 - mask)
        model_in = torch.cat([x, x0_masked, mask], dim=1)
        eps_hat = model(model_in, t_batch)

        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]

        mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_hat)

        if t > 0:
            z = torch.randn_like(x)
            x = mean + torch.sqrt(posterior_variance[t]) * z
        else:
            x = mean

    x = x * mask + x0 * (1.0 - mask)
    return x
