import numpy as np
from PIL import Image

import torch


def pil_to_model_tensor(img: Image.Image, image_size: int, device: torch.device) -> torch.Tensor:
    img = img.convert("RGB").resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # [0,1]
    arr = arr * 2.0 - 1.0  # [-1,1]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def mask_from_canvas_rgba(canvas_rgba: np.ndarray | None, image_size: int, device: torch.device) -> torch.Tensor:
    """Convert RGBA canvas to diffusion mask convention.

    Convention:
    - mask = 1 for pixels to inpaint (missing)
    - mask = 0 for known pixels

    Any painted pixel with alpha>0 is treated as missing.
    """
    if canvas_rgba is None:
        m = np.zeros((image_size, image_size), dtype=np.float32)
    else:
        alpha = canvas_rgba[:, :, 3]
        m = (alpha > 0).astype(np.float32)

    m_t = torch.from_numpy(m)[None, None, :, :].to(device)
    return m_t


def model_tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0
    x = x.squeeze(0).permute(1, 2, 0).numpy()
    x = (x * 255.0).round().astype(np.uint8)
    return Image.fromarray(x)
