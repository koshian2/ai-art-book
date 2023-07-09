import torch
from diffusers import StableDiffusionPipeline
import numpy as np
import torch.nn.functional as F
import modal
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

class DenoisingCallback:
    def __init__(self, initial_value, denoising_threshold):
        self.initial_value = initial_value
        self.denoising_threshold = denoising_threshold

    def callback(self, i, t, latent):
        if t > self.denoising_threshold:
            n = latent.shape[0]
            latent[0:n] = self.initial_value # Avoid call by reference

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "transformers"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def latent_upsclaer(width=1920, height=1024):
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    # Do not use UniPC
    pipe.enable_vae_tiling()
    pipe.to(device)

    prompt = "a girl adventurer is walking alone on a hill by the sea, mountains can be seen in the distance in the background, a small field of flowers, 1girl, blue shirts, green vest, knee-length skirt, blonded-hair, leather hair bands, masterpiece, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    with torch.no_grad():
        generator = torch.Generator().manual_seed(1234)
        noise_large = torch.randn((1,4,height//8,width//8), generator=generator).to(device, torch.float16)
        noise_small = noise_large[:, :, ::2, ::2]
        latent_small = pipe(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=12,
                            latents=noise_small, num_inference_steps=50, output_type="latent").images
        image_small = pipe.numpy_to_pil(pipe.decode_latents(latent_small))[0]
        
        # Denoising callback
        cutoff_threshold = 500 # denoising ratio
        pipe.scheduler.set_timesteps(50, device=device)
        timesteps = pipe.scheduler.timesteps
        threshold_idx = np.where(timesteps.cpu().numpy() <= cutoff_threshold)[0][0]
        latent_large = F.interpolate(latent_small, scale_factor=2, mode="bilinear")
        noised_latents = pipe.scheduler.add_noise(latent_large, noise_large, timesteps)
        mid_latent = noised_latents[threshold_idx:threshold_idx+1]
        warmup_cb = DenoisingCallback(mid_latent, cutoff_threshold)

        image_big = pipe(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=12,
                         latents=noised_latents[0:1], num_inference_steps=50, callback=warmup_cb.callback, output_type="pil").images[0]
        
        image_small.save(f"{CACHE_DIR}/output/11_latent_upscaler_small.jpg", quality=92)
        image_big.save(f"{CACHE_DIR}/output/11_latent_upscaler_big.jpg", quality=92)

if __name__ == "__main__":
    with stub.run():
        latent_upsclaer.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/11_latent_upscaler* .', shell=True)
