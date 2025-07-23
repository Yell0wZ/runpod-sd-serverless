import runpod
from runpod import RunpodLogger
from diffusers import StableDiffusionPipeline
import torch, base64
from io import BytesIO

log = RunpodLogger()
MODEL_ID = "runwayml/stable-diffusion-v1-5"

def load_pipe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe.to(device)
    log.info(f"Model loaded on {device}")
    return pipe

pipe = load_pipe()

def handler(job):
    prompt = job["input"].get("prompt", "a beautiful landscape")
    log.info(f"Prompt: {prompt}")

    image = pipe(prompt, guidance_scale=7.0).images[0]

    buf = BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()

    return {
        "status": "success",
        "image_base64": encoded
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
