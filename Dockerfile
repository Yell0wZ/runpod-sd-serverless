FROM python:3.10

# --- Install OS packages --------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# --- Setup Python env -----------------------------------------------------
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade diffusers transformers peft safetensors accelerate huggingface_hub && \
    pip cache purge && \
    rm -rf /root/.cache/pip

# --- Create model folders -------------------------------------------------
RUN mkdir -p /app/models/lora

# --- Preload flux-diffusion-xl model --------------------------------------
RUN python -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('flux-ml/flux-diffusion-xl', cache_dir='/app/models')"

# --- Copy App Code --------------------------------------------------------
COPY . .

# --- Run App --------------------------------------------------------------
CMD ["python", "-u", "handler.py"]
