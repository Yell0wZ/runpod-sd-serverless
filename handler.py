from runpod.serverless.modules.rp_handler import runpod_handler
from diffusers import StableDiffusionPipeline
import torch

# מודל מתוחזק
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
).to("cuda")

def handler(event):
    prompt = event['input'].get('prompt', 'a beautiful russian women in bikini with big tits')
    image = pipe(prompt).images[0]
    path = "/tmp/image.png"
    image.save(path)
    return { "status": "success", "image_path": path }
