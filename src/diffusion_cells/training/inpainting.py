from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class InpaintingTrainConfig:
    epochs: int = 30
    lr: float = 2e-4
    w_missing: float = 10.0
    grad_clip: float = 1.0
    log_every_steps: int = 50


def random_rect_mask(
    batch_size: int,
    H: int,
    W: int,
    *,
    min_frac: float = 0.2,
    max_frac: float = 0.5,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Random rectangle mask.

    Returns mask of shape (B, 1, H, W) with 1 in missing region to inpaint.
    """
    masks = torch.zeros((batch_size, 1, H, W), device=device)

    for b in range(batch_size):
        rh = int(torch.empty((), device=device).uniform_(min_frac, max_frac).item() * H)
        rw = int(torch.empty((), device=device).uniform_(min_frac, max_frac).item() * W)

        y0 = int(torch.randint(0, max(1, H - rh + 1), (1,), device=device).item())
        x0 = int(torch.randint(0, max(1, W - rw + 1), (1,), device=device).item())

        masks[b, :, y0 : y0 + rh, x0 : x0 + rw] = 1.0

    return masks


def q_sample_batch(
    x0: torch.Tensor,
    t: torch.Tensor,
    sqrt_alpha_bars: torch.Tensor,
    sqrt_one_minus_alpha_bars: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward diffuse a batch: x_t = sqrt(a_bar_t) x0 + sqrt(1-a_bar_t) eps."""
    B = x0.shape[0]
    sa = sqrt_alpha_bars[t].view(B, 1, 1, 1)
    so = sqrt_one_minus_alpha_bars[t].view(B, 1, 1, 1)
    eps = torch.randn_like(x0)
    xt = sa * x0 + so * eps
    return xt, eps


def train_inpainting(
    *,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    T: int,
    alpha_bars: torch.Tensor,
    device: torch.device,
    opt: torch.optim.Optimizer | None = None,
    config: InpaintingTrainConfig = InpaintingTrainConfig(),
    progress: Callable[[int, int, float], None] | None = None,
) -> None:
    """Train eps-prediction UNet for inpainting.

    Mirrors the training loop in notebooks/02_Inpainting_Diffusion.ipynb.

    Args:
        model: UNet accepting (B,7,H,W) + timesteps.
        dataloader: yields x0 in [-1,1] shaped (B,3,H,W).
        T: number of diffusion steps.
        alpha_bars: cumulative product of alphas over T.
        device: torch device.
        opt: optional optimizer (defaults to AdamW with config.lr).
        config: training hyperparameters.
        progress: optional callback (epoch_idx, step_idx, loss_value).
    """
    if opt is None:
        opt = torch.optim.AdamW(model.parameters(), lr=config.lr)

    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    model.train()

    for epoch in range(config.epochs):
        running = 0.0

        for step, x0 in enumerate(dataloader):
            x0 = x0.to(device, non_blocking=True)
            B, _, H, W = x0.shape

            mask = random_rect_mask(B, H, W, min_frac=0.2, max_frac=0.5, device=device)
            x0_masked = x0 * (1.0 - mask)

            t = torch.randint(0, T, (B,), device=device).long()
            xt, eps = q_sample_batch(x0, t, sqrt_alpha_bars, sqrt_one_minus_alpha_bars)

            model_in = torch.cat([xt, x0_masked, mask], dim=1)
            eps_hat = model(model_in, t)

            mse = (eps_hat - eps) ** 2
            mask_c = mask.expand_as(mse)
            loss = (mse * (1.0 + (config.w_missing - 1.0) * mask_c)).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip is not None and config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip))
            opt.step()

            running += float(loss.item())
            if progress is not None and (step % config.log_every_steps == 0):
                progress(epoch, step, running / (step + 1))

        # epoch end callback
        if progress is not None:
            progress(epoch, -1, running / max(1, len(dataloader)))
