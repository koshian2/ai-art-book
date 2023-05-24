import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import numpy as np
from transformers import pipeline, AutoImageProcessor, UperNetForSemanticSegmentation
from PIL import Image

def noise_depth_mask(width=960, height=512):
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    prompts = "masterpiece, best quality, 1girl, brown-haired girl, autumn leaves, standing, beauty, harmony, nature, breathtaking, foliage, dancing, picturesque, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    generator = torch.Generator().manual_seed(1234)
    # latent value
    randn = torch.randn((1, 4, height//8, width//8), generator=generator).to(device, torch.float16)
    # initial run
    initial_image = pipe(prompt=prompts, negative_prompt=negative_prompt, 
                         num_inference_steps=50, output_type="pil", latents=randn).images[0]
    
    depth_estimator = pipeline('depth-estimation')
    depth_map = depth_estimator(initial_image)['depth']
    depth_map = depth_map.resize((width//8, height//8), Image.Resampling.BOX)
    depth_mask = np.array(depth_map) / 255.0 * 0.5 + 0.8
    depth_mask = np.stack([depth_mask for i in range(4)], axis=0)[None]
    masked_randn = randn * torch.from_numpy(depth_mask).to(device, torch.float16)
    # second run
    second_image = pipe(prompt=prompts, negative_prompt=negative_prompt, 
                        num_inference_steps=50, output_type="pil", latents=masked_randn).images[0]