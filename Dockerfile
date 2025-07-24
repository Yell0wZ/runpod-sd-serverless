FROM python:3.10

# --- Install OS packages --------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# --- Setup Python env -----------------------------------------------------
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "diffusers[torch]==0.24.0" transformers peft safetensors accelerate

# --- Create model folders -------------------------------------------------
RUN mkdir -p /app/models/lora

# --- Preload base model and VAE -------------------------------------------
RUN python -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('SG161222/Realistic_Vision_V6.0_B1_noVAE', cache_dir='/app/models')"
RUN python -c "from diffusers import AutoencoderKL; AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse-original', cache_dir='/app/models')"

# --- Copy App Code --------------------------------------------------------
COPY . .

# --- Run App --------------------------------------------------------------
CMD ["python", "-u", "handler.py"]
