# Use a lightweight Python base image
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable stdout buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/src \
    PIP_NO_CACHE_DIR=1

# Install system deps needed for Pillow and canvas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Run from root
WORKDIR /

# Copy project to root
COPY . /

# Copy only the required checkpoint into the image
COPY checkpoints/phase2_inpaint_unet.pt /checkpoints/phase2_inpaint_unet.pt

# Install Python deps
RUN pip install --upgrade pip setuptools wheel && \
    pip install -U streamlit streamlit-drawable-canvas-fix numpy pillow && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Expose Streamlit default port
EXPOSE 8501

# Streamlit config to suppress analytics prompts
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Default command: run the main app (you can change to app_test.py if needed)
CMD ["streamlit", "run", "/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
