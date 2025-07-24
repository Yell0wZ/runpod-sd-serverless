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

# --- Preload FLUX model ---------------------------------------------------
RUN python -c "from diffusers import FluxPipeline; FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-schnell', cache_dir='/app/models')"

# --- Copy App Code --------------------------------------------------------
COPY . .

# --- Run App --------------------------------------------------------------
CMD ["python", "-u", "handler.py"]
