import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

prompt = "realistic photo frog"
images = pipe(prompt).images

image = images[0]
    
image.save("jake_images/frog.png")