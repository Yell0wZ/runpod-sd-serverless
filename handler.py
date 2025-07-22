from runpod.serverless.modules.rp_handler import runpod_handler
from diffusers import StableDiffusionPipeline
import torch
import base64

# Load model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
).to("cuda")

@runpod_handler
def handler(event):
    prompt = event['input'].get('prompt', 'a beautiful russian woman in bikini with big tits')
    image = pipe(prompt).images[0]
    
    # Save to temp file
    path = "/tmp/image.png"
    image.save(path)

    # Read file as base64
    with open(path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "status": "success",
        "prompt": prompt,
        "image_base64": image_b64
    }
