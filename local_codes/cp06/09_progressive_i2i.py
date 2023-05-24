from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image

def progressive_i2i(width=1920, height=1024):
    device = "cuda"    
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    prompt = "a girl adventurer is walking alone on a hill by the sea, mountains can be seen in the distance in the background, a small field of flowers, 1girl, blue shirts, green vest, knee-length skirt, blonded-hair, leather hair bands, masterpiece, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, " \
                    "fewer digits, cropped, worst quality, low quality"

    generator = torch.Generator().manual_seed(1234)
    # latent value
    randn = torch.randn((1, 4, height//16, width//16), generator=generator).to(device, torch.float16)
    # initial run
    initial_image = pipe(prompt=prompt, negative_prompt=negative_prompt, 
                         num_inference_steps=50, output_type="pil", latents=randn, guidance_scale=12).images[0]
    
    # upsampling
    upsampled_image = initial_image.resize((initial_image.width*2, initial_image.height*2), Image.Resampling.BICUBIC)

    pipe = StableDiffusionImg2ImgPipeline(**pipe.components)
    second_image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=upsampled_image,
                         num_inference_steps=30, output_type="pil", strength=0.7).images[0]
