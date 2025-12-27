import io, sys
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from dataclasses import dataclass

import torch
import torch.nn as nn

# The project modules:
# Ensure local package path is available
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from diffusion_cells.diffusion import ddpm_inpaint_noised_clamp, linear_beta_schedule
from diffusion_cells.io import mask_from_canvas_rgba, model_tensor_to_pil, pil_to_model_tensor
from diffusion_cells.models import UNetSmall



# -----------------------------
# Model paths
# -----------------------------

checkpoint_path = "checkpoints/phase2_inpaint_unet.pt"


# -----------------------------
# Pre/post-processing
# -----------------------------

@dataclass(frozen=True)
class AppConfig:
    image_size: int = 64
    time_dim: int = 256
    base_ch: int = 64
    in_ch: int = 7


@st.cache_resource
def load_model(checkpoint_path: str, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})

    model = UNetSmall(
        in_ch=int(cfg.get("in_ch", 7)),
        base_ch=int(cfg.get("base_ch", 64)),
        time_dim=int(cfg.get("time_dim", 256)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, cfg


st.set_page_config(page_title="Diffusion Inpainting", layout="wide")
st.title("Diffusion inpainting app")

# Sidebar settings to match app.py look (own background and grouping)
with st.sidebar:
    st.header("Settings")
    brush_size = st.slider("Brush size", min_value=2, max_value=64, value=16)
    brush_opacity = st.slider("Brush opacity", min_value=0.05, max_value=1.0, value=0.45)
    steps = st.number_input("Diffusion steps (T)", min_value=50, max_value=2000, value=1000, step=50)
    beta_start = st.number_input("beta_start", min_value=1e-6, max_value=1e-2, value=1e-4, format="%.6f")
    beta_end = st.number_input("beta_end", min_value=1e-4, max_value=1e-1, value=2e-2, format="%.6f")

# Right column: upload + brushable image
right = st.columns([1])[0]
with right:
    st.subheader("Step 1: Upload image")
    uploaded = st.file_uploader("Upload an RGB image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

    if uploaded is None:
        st.info("Upload an image to begin.")
        st.stop()

    img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

    # Optional: cap width for nicer UX
    max_w = 900
    if img.width > max_w:
        scale = max_w / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)))

    # Draw directly over the uploaded image using it as the canvas background
    st.subheader("Step 2: Draw mask - paint the region to inpaint")
    st.caption("Mask convention: painted area = missing (=1), unpainted = known (=0).")
    canvas = st_canvas(
        fill_color=f"rgba(255, 255, 255, {brush_opacity})",
        stroke_width=brush_size,
        stroke_color=f"rgba(255, 0, 0, {brush_opacity})",
        background_image=img,
        update_streamlit=True,
        height=img.height,
        width=img.width,
        drawing_mode="freedraw",
        key="canvas",
    )


if canvas.image_data is None:
    st.stop()

rgba = canvas.image_data.astype(np.uint8)
alpha = rgba[..., 3]
mask = (alpha > 0).astype(np.uint8) * 255

# Conditioning preview like app.py
img_arr = np.array(img).astype(np.float32)
masked_img = img_arr.copy()
masked_img[mask == 255] = 0
masked_img = masked_img.astype(np.uint8)


# Download masked conditioning image
buf = io.BytesIO()
Image.fromarray(masked_img).save(buf, format="PNG")
st.download_button(
    "Download masked image (conditioning PNG)",
    data=buf.getvalue(),
    file_name="masked_conditioning.png",
    mime="image/png",
)

# Add primary Run button directly under the canvas
run = st.button("Run inpainting")

# If user clicked run, this is where you'd trigger inference
if run:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare 64×64 tensors for image
    x0_64 = pil_to_model_tensor(img, 64, device)
    # Resize canvas alpha to 64×64 for a proper binary mask
    alpha_rgba = canvas.image_data.astype(np.uint8)
    alpha_pil = Image.fromarray(alpha_rgba).split()[3]
    alpha_64 = alpha_pil.resize((64, 64), resample=Image.Resampling.NEAREST)
    mask_np = (np.array(alpha_64) > 0).astype(np.float32)
    mask_t64 = torch.from_numpy(mask_np).to(device).unsqueeze(0).unsqueeze(0)
    # Broadcast mask to match x0 channels in later ops if needed
    x0_masked_64 = x0_64 * (1.0 - mask_t64)
    preview_64 = model_tensor_to_pil(x0_masked_64)
    #st.caption("64×64 masked conditioning (preprocessed)")
    #st.image(preview_64, use_container_width=False)

    # Load model
    try:
        model, cfg = load_model(checkpoint_path, device)
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        #return

    # Build schedule (must match training to be faithful; default matches notebook)
    T = int(steps)
    betas, alphas, alpha_bars = linear_beta_schedule(T, beta_start=float(beta_start), beta_end=float(beta_end), device=device)
    alpha_bars_prev = torch.cat([torch.tensor([1.0], device=device), alpha_bars[:-1]])
    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

    with st.spinner("Sampling..."):
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

        ### Simple display of generated result resized to match displayed image:
        #out_img = model_tensor_to_pil(x_gen)
        ## Resize inpainted result to the same size as the displayed image
        #out_img_resized = out_img.resize((img.width, img.height), resample=Image.Resampling.BILINEAR)
        #st.caption("Inpainted result")
        #st.image(out_img_resized, width=img.width, use_container_width=False)
        ### or composited onto original image size:

        out_img_64 = model_tensor_to_pil(x_gen)
        # Upscale generated result to match the displayed image size
        out_img_hr = out_img_64.resize((img.width, img.height), resample=Image.Resampling.BILINEAR)
        # Build high-res binary mask from canvas alpha (True where to replace)
        alpha_hr = Image.fromarray(canvas.image_data.astype(np.uint8)).split()[3]
        mask_hr = (np.array(alpha_hr) > 0)
        # Composite: replace only masked pixels
        orig_np = np.array(img)
        gen_np = np.array(out_img_hr)
        comp_np = orig_np.copy()
        comp_np[mask_hr] = gen_np[mask_hr]
        comp_img = Image.fromarray(comp_np)
        st.caption("Inpainted result (composited onto original image size)")
        st.image(comp_img, width=img.width, use_container_width=False)
        # Download composite
        _buf = io.BytesIO()
        comp_img.save(_buf, format="PNG")
        st.download_button("Download inpainted composite (PNG)", data=_buf.getvalue(), file_name="inpainted_composite.png", mime="image/png")


