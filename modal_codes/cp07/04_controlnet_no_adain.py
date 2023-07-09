
import torch
from diffusers import UniPCMultistepScheduler
from stable_diffusion_reference import StableDiffusionReferencePipeline
from PIL import Image
import matplotlib.pyplot as plt
import modal
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]==0.16.1", "transformers", "matplotlib"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    timeout=600,
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def main():
    initial_img = Image.open("data/black_hair_girl.png")

    device = "cuda"
    pipe = StableDiffusionReferencePipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16, safety_checker=None, cache_dir=CACHE_DIR)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_tiling()
    pipe.to(device, torch.float16)

    prompt = "best quality"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    fig = plt.figure(figsize=(12, 12))
    for fi, fidelity in enumerate([0, 0.5, 1]):
        generator = torch.Generator().manual_seed(1234)
        image = pipe(prompt=prompt, ref_image=initial_img, negative_prompt=negative_prompt, style_fidelity=fidelity, reference_adain=False,
                    generator=generator, num_inference_steps=30, num_images_per_prompt=5).images

        for i in range(5):
            ax = fig.add_subplot(3, 5, fi*5+i+1)
            ax.imshow(image[i])
            ax.axis("off")
            if i == 0:
                ax.set_title(f"style_fidelity={fidelity}")

    fig.savefig(f"{CACHE_DIR}/output/04_controlnet_no_adain.png")

if __name__ == "__main__":
    with stub.run():
        main.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/04_controlnet_no_adain.png output', shell=True)
