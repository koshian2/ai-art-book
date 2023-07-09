from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch
import modal
import os
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

@stub.function(
    image=modal.Image.debian_slim().pip_install("torch", "diffusers[torch]", "transformers", "accelerate"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def run_stable_diffusion(prompt,
                         output_filename,
                         model_name="stabilityai/stable-diffusion-2-1-base",
                         seed=1234):
    device="cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_tiling()
    pipe.to(device)

    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    generator = torch.Generator(device).manual_seed(seed)
    image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=30, generator=generator).images[0]

    output_filename = f"{CACHE_DIR}/{output_filename}"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    image.save(output_filename)

def main():
    with stub.run():
        output_file = "output/astro_cat_modal.png"
        run_stable_diffusion.call("Cat in outer space, look at viewer, best quality",
                                  output_file, seed=1235)
        subprocess.run(
            f'modal nfs get model-cache-vol {output_file} {output_file}', shell=True)
        
if __name__ == "__main__":
    main()
