import runpod
from runpod.serverless.modules.rp_logger import RunPodLogger
from diffusers import StableDiffusionPipeline
import torch, base64
from io import BytesIO

log = RunPodLogger()

pipe = StableDiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V6.0_B1_noVAE",
    cache_dir="/app/models",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")

pipe.safety_checker = None
pipe.requires_safety_checker = False

def handler(job):
    prompt = job["input"].get("prompt", "a beautiful landscape")
    lora_model = job["input"].get("lora_model", None)
    lora_scale = job["input"].get("lora_scale", 0.8)
    
    log.info(f"Prompt: {prompt}")
    
    # Load LoRA if specified
    if lora_model:
        try:
            pipe.load_lora_weights(lora_model)
            log.info(f"Loaded LoRA: {lora_model}")
        except Exception as e:
            log.error(f"Failed to load LoRA {lora_model}: {e}")

    image = pipe(
        prompt, 
        num_inference_steps=30,
        guidance_scale=7.5,
        negative_prompt = "cartoon,  drawing, CGI, 3D, plastic, blurry, painting, unrealistic",
        height=768,
        width=512,
        cross_attention_kwargs={"scale": lora_scale} if lora_model else {}
    ).images[0]
    
    # Unload LoRA after generation
    if lora_model:
        try:
            pipe.unload_lora_weights()
        except:
            pass
    buf = BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()

    return {"status": "success", "image_base64": encoded}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
