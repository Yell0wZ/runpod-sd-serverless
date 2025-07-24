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
    log.info(f"Prompt: {prompt}")

    image = pipe(
        prompt, 
        num_inference_steps=30,
        guidance_scale=7.0,
        negative_prompt="cartoon, anime, painting, drawing, illustration, digital art"
    ).images[0]
    buf = BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()

    return {"status": "success", "image_base64": encoded}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
