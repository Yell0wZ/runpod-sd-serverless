import runpod
from runpod.serverless.modules.rp_logger import RunPodLogger
from diffusers import StableDiffusionPipeline
import torch, base64
from io import BytesIO

log = RunPodLogger()

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    cache_dir="/app/models",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")

pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

def handler(job):
    prompt = job["input"].get("prompt", "a beautiful landscape")
    log.info(f"Prompt: {prompt}")

    image = pipe(prompt).images[0]
    buf = BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()

    return {"status": "success", "image_base64": encoded}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
