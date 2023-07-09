
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
    timeout=600,
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def main():
    initial_img = Image.open("data/texture_leaf.jpg")

    device = "cuda"
    model_id = "NoCrypt/SomethingV2_2"
    pipe = StableDiffusionReferencePipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None, cache_dir=CACHE_DIR)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device, torch.float16)

    prompt = "silver-colored ring, diamond, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    fig = plt.figure(figsize=(18, 7))
    for fi, use_reference_attn in enumerate([False, True]):
        generator = torch.Generator().manual_seed(1234)
        image = pipe(prompt=prompt, ref_image=initial_img, negative_prompt=negative_prompt, 
                     style_fidelity=0.0, reference_adain=True,
                     reference_attn=use_reference_attn,
                    generator=generator, num_inference_steps=30, num_images_per_prompt=4).images

        for i in range(4):
            ax = fig.add_subplot(2, 4, fi*4+i+1)
            ax.imshow(image[i])
            ax.axis("off")
            if i == 0:
                ax.set_title(f"Reference Attn = {use_reference_attn}")
    fig.savefig(f"{CACHE_DIR}/output/07_controlnet_with_adain_no_attention.png")

if __name__ == "__main__":
    with stub.run():
        main.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/07_controlnet_with_adain_no_attention.png output', shell=True)
