#!/usr/bin/env python3

# It keeps the UI running on a small CPU instance 
# while spinning up a GPU only when the user clicks “Run inpainting.”

import os
import io
import json
from pathlib import Path

import boto3
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

# Ensure local package path is available
import sys
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from diffusion_cells.diffusion import ddpm_inpaint_noised_clamp, linear_beta_schedule
from diffusion_cells.io import model_tensor_to_pil, pil_to_model_tensor
from diffusion_cells.models import UNetSmall


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = UNetSmall(
        in_ch=int(cfg.get("in_ch", 7)),
        base_ch=int(cfg.get("base_ch", 64)),
        time_dim=int(cfg.get("time_dim", 256)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model


def read_s3_image(s3, uri: str) -> Image.Image:
    assert uri.startswith("s3://"), "INPUT_S3_URI must start with s3://"
    _, _, rest = uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")


def read_s3_mask(s3, uri: str) -> np.ndarray:
    # Mask image should be single-channel with white where masked; same resolution as uploaded image
    im = read_s3_image(s3, uri).convert("L")
    arr = np.array(im)
    return (arr > 0).astype(np.uint8)


def write_s3_image(s3, uri: str, img: Image.Image) -> None:
    _, _, rest = uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue(), ContentType="image/png")


def main():
    input_uri = os.environ.get("INPUT_S3_URI")
    mask_uri = os.environ.get("MASK_S3_URI")
    output_uri = os.environ.get("OUTPUT_S3_URI")
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "/checkpoints/phase2_inpaint_unet.pt")
    steps = int(os.environ.get("STEPS", "1000"))
    beta_start = float(os.environ.get("BETA_START", "1e-4"))
    beta_end = float(os.environ.get("BETA_END", "2e-2"))

    if not (input_uri and mask_uri and output_uri):
        raise RuntimeError("Missing required env vars: INPUT_S3_URI, MASK_S3_URI, OUTPUT_S3_URI")

    s3 = boto3.client("s3")

    # Read inputs
    img_hr = read_s3_image(s3, input_uri)
    mask_hr = read_s3_mask(s3, mask_uri)  # HxW, 1=masked

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare 64×64 tensors
    x0_64 = pil_to_model_tensor(img_hr, 64, device)
    # Downscale mask to 64×64 (nearest)
    mask_pil = Image.fromarray((mask_hr * 255).astype(np.uint8), mode="L").resize((64, 64), resample=Image.Resampling.NEAREST)
    mask_np_64 = (np.array(mask_pil) > 0).astype(np.float32)
    mask_t64 = torch.from_numpy(mask_np_64).to(device).unsqueeze(0).unsqueeze(0)

    # Load model
    model = load_model(checkpoint_path, device)

    # Schedule
    T = steps
    betas, alphas, alpha_bars = linear_beta_schedule(T, beta_start=beta_start, beta_end=beta_end, device=device)
    alpha_bars_prev = torch.cat([torch.tensor([1.0], device=device), alpha_bars[:-1]])
    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

    # Sample
    x_gen = ddpm_inpaint_noised_clamp(
        model=model,
        x0=x0_64,
        mask=mask_t64,
        T=T,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        posterior_variance=posterior_variance,
        device=device,
    )

    out_64 = model_tensor_to_pil(x_gen)

    # Upscale to original and composite only masked region
    out_hr = out_64.resize((img_hr.width, img_hr.height), resample=Image.Resampling.BILINEAR)
    orig_np = np.array(img_hr)
    gen_np = np.array(out_hr)
    comp_np = orig_np.copy()
    comp_np[mask_hr.astype(bool)] = gen_np[mask_hr.astype(bool)]
    comp_img = Image.fromarray(comp_np)

    write_s3_image(s3, output_uri, comp_img)

    # Optional: emit a small JSON status artifact
    status_uri = os.environ.get("STATUS_S3_URI")
    if status_uri:
        _, _, rest = status_uri.partition("s3://")
        bucket, _, key = rest.partition("/")
        s3.put_object(Bucket=bucket, Key=key, Body=json.dumps({"status": "done"}).encode("utf-8"), ContentType="application/json")


if __name__ == "__main__":
    main()
