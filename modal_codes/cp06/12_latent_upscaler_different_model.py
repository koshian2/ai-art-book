import torch
from diffusers import StableDiffusionPipeline
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
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

def run_unit(pipe, prompt, device="cuda", width=1920, height=1024):
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    with torch.no_grad():
        generator = torch.Generator().manual_seed(1234)
        noise_large = torch.randn((1,4,height//8,width//8), generator=generator).to(device, torch.float16)
        noise_small = noise_large[:, :, ::2, ::2]
        latent_small = pipe(prompt=prompt, negative_prompt=negative_prompt,
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

        image_big = pipe(prompt=prompt, negative_prompt=negative_prompt,
                         latents=noised_latents[0:1], num_inference_steps=50, callback=warmup_cb.callback, output_type="pil").images[0]
        return image_small, image_big

def merge_network(pipe_source, pipe_merge, attr, ratio):
    merge_net = copy.deepcopy(getattr(pipe_source, attr))
    pipe_source_params = dict(getattr(pipe_source, attr).named_parameters())
    pipe_merge_params = dict(getattr(pipe_merge, attr).named_parameters())
    for key, param in merge_net.named_parameters():
        x = pipe_source_params[key] * (1-ratio) + pipe_merge_params[key] * ratio
        param.data = x
    return merge_net

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "transformers", "matplotlib"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="a10g",
    timeout=600,
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def main():
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe.enable_vae_tiling()
    pipe.to(device)

    results = []
    results += run_unit(pipe, 
                        "a girl standing on the beach, smiling and having fun, blue haired, 1girl, border skirt, denim jacket, best quality, extremely detailed")

    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe_pastel = StableDiffusionPipeline.from_pretrained("JamesFlare/pastel-mix",
                                                           torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe = StableDiffusionPipeline.from_pretrained(
        "prompthero/openjourney-v4", torch_dtype=torch.float16, cache_dir=CACHE_DIR,
        safety_checker=None)
    pipe.unet = merge_network(pipe, pipe_pastel, "unet", 0.75)
    pipe.text_encoder = merge_network(pipe, pipe_pastel, "text_encoder", 0.75)
    pipe.vae = sd_pipe.vae
    pipe.enable_vae_tiling()
    pipe.to(device)

    results += run_unit(pipe,
                        "brave female knight, dragon, intense battle, reddish-brown short hair, blue determined eyes, silver armor, beautiful design, emblem, red jacket, leather boots, large sword, green and black scales, formidable dragon, rocky battlefield, old stone walls, flames, dark clouds, raindrops, sword and scale reflections, dramatic scene, best quality, extremely detailed")

    fig = plt.figure(figsize=(18, 10))
    titles = ["[1] Low-Res", "[1] High-Res", "[2] Low-Res", "[2] High-Res"]
    for i, (title, img) in enumerate(zip(titles, results)):
        ax = fig.add_subplot(2, 2, i+1)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title)
    fig.savefig(f"{CACHE_DIR}/output/12_latent_upscaler_different_model.png")

if __name__ == "__main__":
    with stub.run():
        main.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/12_latent_upscaler_different_model.png output', shell=True)
