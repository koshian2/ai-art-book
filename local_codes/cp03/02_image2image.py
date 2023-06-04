from diffusers import StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler
from PIL import Image
import torch

def run_image2image(prompt,
                    base_image,
                    output_filename, 
                    model_name="stabilityai/stable-diffusion-2-1-base",
                    device="cuda", seed=1234):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_tiling()
    pipe.to(device)

    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    with Image.open(base_image) as img:
        generator = torch.Generator(device).manual_seed(seed)
        image = pipe(prompt, negative_prompt=negative_prompt, image=img,
                     num_inference_steps=30, generator=generator).images[0]
        image.save(output_filename)

if __name__ == "__main__":
    run_image2image("Cat that landed on Mars, look at viewer, best quality",
                    base_image="output/astro_cat2.png",
                    output_filename="output/astro_mars.png",
                    model_name="NoCrypt/SomethingV2_2", seed=1235)
