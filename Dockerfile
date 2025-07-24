FROM python:3.10

# --- OS packages -----------------------------------------------------------
RUN apt-get update && apt-get install -y \
        git \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# --- Python env ------------------------------------------------------------
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Pre-download Stable Diffusion model ----------------------------------
RUN python -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('SG161222/Realistic_Vision_V6.0_B1_noVAE', cache_dir='/app/models', use_safetensors=True)"

# --- App code --------------------------------------------------------------
COPY . .

# --- Run the handler (‑u = unbuffered stdout for clear logs) ---------------
CMD ["python", "-u", "handler.py"]
