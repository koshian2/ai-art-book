from diffusers import StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler
from PIL import Image
import torch
import modal
import os
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

@stub.function(
    image=modal.Image.debian_slim().pip_install("torch", "diffusers[torch]", "transformers"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def run_image2image(prompt,
                    base_image,
                    output_filename,
                    model_name="stabilityai/stable-diffusion-2-1-base",
                    device="cuda", seed=1234):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_tiling()
    pipe.to(device)

    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    with Image.open(base_image) as img:
        generator = torch.Generator(device).manual_seed(seed)
        image = pipe(prompt, negative_prompt=negative_prompt, image=img,
                     num_inference_steps=30, generator=generator).images[0]

    output_filename = f"{CACHE_DIR}/{output_filename}"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    image.save(output_filename)

def main(prompt,
         base_image,
         output_filename,
         model_name="stabilityai/stable-diffusion-2-1-base", seed=1234):
    with stub.run():
        run_image2image.call(prompt, base_image, output_filename, model_name, seed=seed)
        subprocess.run(
            f'modal nfs get model-cache-vol {output_filename} {output_filename}', shell=True)

if __name__ == "__main__":
    main("Cat that landed on Mars, look at viewer, best quality",
            base_image="output/astro_cat2.png",
            output_filename="output/astro_mars.png",
            model_name="NoCrypt/SomethingV2_2", seed=1235)
