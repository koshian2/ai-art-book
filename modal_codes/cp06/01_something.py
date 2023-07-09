import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import os
import modal
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

class WarmupCallback:
    def __init__(self, initial_value, denoising_threshold):
        self.initial_value = initial_value
        self.denoising_threshold = denoising_threshold

    def callback(self, i, t, latent):
        if t > self.denoising_threshold:
            n = latent.shape[0]
            latent[0:n] = self.initial_value # Avoid 

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "transformers"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def main(width=960, height=512):
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    generator = torch.Generator().manual_seed(1234)
    prompts = "a girl standing on the beach, smiling and having fun, blue haired, 1girl, border skirt, denim jacket, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    image = pipe(prompt=prompts, negative_prompt=negative_prompt, generator=generator,
                 num_inference_steps=50, output_type="pil", width=width, height=height).images[0]
    os.makedirs(f"{CACHE_DIR}/output", exist_ok=True)
    image.save(f"{CACHE_DIR}/output/01_base.png")
    
if __name__ == "__main__":
    with stub.run():
        main.call()
    os.makedirs("output", exist_ok=True)
    subprocess.run(
        f'modal nfs get model-cache-vol output/01_base.png output', shell=True)
