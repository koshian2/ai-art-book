from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch
import matplotlib.pyplot as plt
from utils import LORA_PATH, load_safetensors_lora
from huggingface_hub import hf_hub_download
import modal
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

def download_lora():
    for download_file in ["bigeye1.safetensors", "flat2.safetensors"]:
        hf_hub_download(repo_id="2vXpSwA7/iroiro-lora", 
                        filename=download_file,
                        subfolder="release",
                        local_dir=LORA_PATH)
@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "transformers", "matplotlib"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def run_flat_lora(prompt,
                  model_name="stabilityai/stable-diffusion-2-1-base",
                  device="cuda", seed=1234):
    download_lora()

    fig = plt.figure(figsize=(18, 10))
    for i, lora_name in enumerate(["bigeye1.safetensors", "flat2.safetensors"]):
        alphas = [-1, -0.5, 0, 0.5, 1]
        for j, alpha in enumerate(alphas):
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name, torch_dtype=torch.float16, safety_checker=None)
            pipe = load_safetensors_lora(pipe, f"{LORA_PATH}/release/{lora_name}", alpha=alpha)

            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.vae.enable_tiling()
            pipe.to(device)

            negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

            generator = torch.Generator(device).manual_seed(seed)
            image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=30, 
                        generator=generator, height=768).images[0]
            
            ax = fig.add_subplot(2, 5, i*(len(alphas))+j+1)
            ax.imshow(image)
            ax.set_title(f"{lora_name} | α={alpha}")
            ax.axis("off")

    fig.savefig(f"{CACHE_DIR}/output/03_flat_lora.png")

if __name__ == "__main__":
    with stub.run():
        run_flat_lora.call("1girl, look at viewer, best quality", 
                    model_name=f"{CACHE_DIR}/Counterfeit-V3.0")
    subprocess.run(
        f'modal nfs get model-cache-vol output/03_flat_lora.png output', shell=True)
