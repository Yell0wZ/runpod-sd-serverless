from runpod.serverless.modules.rp_handler import runpod_handler
from diffusers import StableDiffusionPipeline
import torch
import base64
from io import BytesIO

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
).to("cuda")

@runpod_handler
def handler(event):
    prompt = event['input'].get('prompt', 'a beautiful russian woman in bikini with big tits')
    image = pipe(prompt).images[0]

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "status": "success",
        "prompt": prompt,
        "image_base64": encoded_image
    }
