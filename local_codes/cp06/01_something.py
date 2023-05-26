import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import modal
import io
import numpy as np
import torch.nn.functional as F
from PIL import Image

stub = modal.Stub()
CACHE_PATH = "/root/model_cache"
volume = modal.SharedVolume().persist("stable-diff-model-vol")

class WarmupCallback:
    def __init__(self, initial_value, denoising_threshold):
        self.initial_value = initial_value
        self.denoising_threshold = denoising_threshold

    def callback(self, i, t, latent):
        if t > self.denoising_threshold:
            n = latent.shape[0]
            latent[0:n] = self.initial_value # Avoid 

@stub.function(
    image=modal.Image.debian_slim().pip_install("torch", "diffusers", "transformers", "accelerate", "safetensors"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    shared_volumes={CACHE_PATH: volume},
    gpu="t4",
)
def main(width=960, height=512):
    device = "cuda"
    dtype = torch.float16 if "cuda" in device else torch.float32    
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", cache_dir=CACHE_PATH, torch_dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    generator = torch.Generator().manual_seed(1234)
    prompts = "a girl standing on the beach, smiling and having fun, blue haired, 1girl, border skirt, denim jacket, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    image = pipe(prompt=prompts, negative_prompt=negative_prompt, generator=generator,
                 num_inference_steps=50, output_type="pil", width=width, height=height).images[0]
    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        return buf.getvalue()

if __name__ == "__main__":
    with stub.run():
        result = main.call()
        with open("output/05/01_base.png", "wb") as f:
            f.write(result)
