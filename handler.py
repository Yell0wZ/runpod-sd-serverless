from diffusers import FluxPipeline
import torch, base64, runpod, os
from runpod.serverless.modules.rp_logger import RunPodLogger
from io import BytesIO
from huggingface_hub import login

log = RunPodLogger()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Authenticate with Hugging Face if token is available
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    log.info("Authenticated with Hugging Face")
else:
    log.warning("No HF_TOKEN found - FLUX model may not load")

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    cache_dir="/app/models",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
).to(device)

pipe.safety_checker = None
pipe.requires_safety_checker = False

def handler(job):
    prompt      = job["input"].get("prompt", "RAW photo of a woman, dslr")
    lora_path   = job["input"].get("lora_model")   # ‎/app/models/lora/my.safetensors
    lora_scale  = float(job["input"].get("lora_scale", 0.8))

    if lora_path:
        pipe.load_lora_weights(lora_path, adapter_name="dyn")
        log.info(f"Loaded LoRA {lora_path}")

    img = pipe(
        prompt,
        height=1024, width=1024,               # FLUX optimal resolution
        num_inference_steps=4,                 # FLUX works best with 4 steps
        guidance_scale=0.0,                    # FLUX doesn't use guidance scale
        max_sequence_length=256,               # FLUX optimal sequence length
        generator=torch.Generator("cpu").manual_seed(42)
    ).images[0]

    if lora_path:
        pipe.unload_lora_weights()             # משחרר VRAM

    buf = BytesIO(); img.save(buf, format="PNG")
    return {"status": "success",
            "image_base64": base64.b64encode(buf.getvalue()).decode()}


runpod.serverless.start({"handler": handler})
