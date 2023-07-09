import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import matplotlib.pyplot as plt
import modal
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "transformers", "matplotlib"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def noise_scale(width=960, height=512):
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    result = []
    # Guidance Scale by specifing latent
    prompts = "great panorama of snowy mountains, best quality, extremely detailed"
    negative_prompt = "worst quality, low quality"
    for gs in [1.5, 7, 10, 20, 50, 100]:
        generator = torch.Generator().manual_seed(1234)
        # Latentを直接指定してもいける
        latent = torch.randn((1, 4, height//8, width//8), generator=generator).to(device, torch.float16)
        image = pipe(prompt=prompts, negative_prompt=negative_prompt, guidance_scale=gs,
                    num_inference_steps=50, output_type="pil", latents=latent).images[0]
        result.append({"mode": "Guidance Scale", "value": gs, "pic": image})

    for scale in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        generator = torch.Generator().manual_seed(1234)
        # Noise scale
        latent = torch.randn((1, 4, height//8, width//8), generator=generator).to(device, torch.float16) * scale
        image = pipe(prompt=prompts, negative_prompt=negative_prompt,
                    num_inference_steps=50, output_type="pil", latents=latent).images[0]
        result.append({"mode": "Noise Scale", "value": scale, "pic": image})

    fig = plt.figure(figsize=(16, 12))
    for i, item in enumerate(result):
        ax = fig.add_subplot(4, 3, i+1)
        ax.imshow(item["pic"])
        ax.axis("off")
        ax.set_title(f"{item['mode']}={item['value']}")
    fig.savefig(f"{CACHE_DIR}/output/05_noise_scale.png")

if __name__ == "__main__":
    with stub.run():
        noise_scale.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/05_noise_scale.png output', shell=True)    
