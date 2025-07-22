from runpod.serverless.modules.rp_handler import runpod_handler
from diffusers import StableDiffusionPipeline
import torch

# Load model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
).to("cuda")

@runpod_handler
def handler(event):
    prompt = event['input'].get('prompt', 'a beautiful russian woman in bikini with big tits')
    image = pipe(prompt).images[0]
    path = "/tmp/image.png"
    image.save(path)
    return { "status": "success", "image_path": path }
