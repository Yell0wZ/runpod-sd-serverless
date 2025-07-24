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
RUN pip install --no-cache-dir --upgrade diffusers transformers peft safetensors accelerate huggingface_hub

# --- Create model folders -------------------------------------------------
RUN mkdir -p /app/models/lora

# --- Note: FLUX model will be downloaded at runtime due to size ---

# --- Copy App Code --------------------------------------------------------
COPY . .

# --- Run App --------------------------------------------------------------
CMD ["python", "-u", "handler.py"]
