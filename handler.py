from diffusers import StableDiffusionPipeline
import torch, base64, runpod
from runpod.serverless.modules.rp_logger import RunPodLogger
from io import BytesIO

log = RunPodLogger()
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    "HiDream-ai/HiDream-I1-Dev",
    cache_dir="/app/models",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    use_safetensors=True
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
        height=768, width=768,                 # Standard SD resolution
        num_inference_steps=25,
        guidance_scale=7.0,
        negative_prompt="blurry, low quality, distorted, deformed, ugly",
        generator=torch.Generator("cpu").manual_seed(42)
    ).images[0]

    if lora_path:
        pipe.unload_lora_weights()             # משחרר VRAM

    buf = BytesIO(); img.save(buf, format="PNG")
    return {"status": "success",
            "image_base64": base64.b64encode(buf.getvalue()).decode()}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
