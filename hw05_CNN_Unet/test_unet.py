from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline

# Image-to-Image text-guided generation with Stable Diffusion
# Download model https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main
# Use a 3090/4090/A100 to perform the following experiments

model_id_or_path = "/data/LLM_MODEL/stabilityai/stable-diffusion-v1-4"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# check how many CrossAttnDownBlock2D/CrossAttnUpBlock2D blobks in the unet.
# what is mid-block? https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_condition.py

print(pipe.unet)


init_image = Image.open('Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg').convert("RGB")
init_image = init_image.resize((768, 512))

prompt = "A fantasy landscape, trending on artstation"

images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images

images[0].save("fantasy_landscape.png")