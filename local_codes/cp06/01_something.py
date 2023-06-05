import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import os

class WarmupCallback:
    def __init__(self, initial_value, denoising_threshold):
        self.initial_value = initial_value
        self.denoising_threshold = denoising_threshold

    def callback(self, i, t, latent):
        if t > self.denoising_threshold:
            n = latent.shape[0]
            latent[0:n] = self.initial_value # Avoid 

def main(width=960, height=512):
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    generator = torch.Generator().manual_seed(1234)
    prompts = "a girl standing on the beach, smiling and having fun, blue haired, 1girl, border skirt, denim jacket, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    image = pipe(prompt=prompts, negative_prompt=negative_prompt, generator=generator,
                 num_inference_steps=50, output_type="pil", width=width, height=height).images[0]
    os.makedirs("output", exist_ok=True)
    image.save("output/01_base.png")
    
if __name__ == "__main__":
    main()
