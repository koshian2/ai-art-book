import torch
from diffusers import DiffusionPipeline, UniPCMultistepScheduler
from utils import load_safetensors_lora
from settings import LORA_DIRECTORY
import modal
import subprocess
import requests
import os

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

# settings.pyにLoRAを保存したディレクトリを記述
def download_lora():
    urlData = requests.get("https://civitai.com/api/download/models/41292").content
    os.makedirs(LORA_DIRECTORY, exist_ok=True)
    with open(f"{LORA_DIRECTORY}/lumine1-000008.safetensors" ,mode='wb') as fp:
        fp.write(urlData)

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "transformers"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def run_lora(width=512, height=960):
    download_lora()

    device = "cuda"    
    pipe = DiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16,
         custom_pipeline="lpw_stable_diffusion", cache_dir=CACHE_DIR)
    pipe = load_safetensors_lora(pipe, f"{LORA_DIRECTORY}/lumine1-000008.safetensors", alpha=0.5)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    prompt = "luminedef, luminernd, 1girl, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    generator = torch.Generator(device).manual_seed(1234)
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, max_embeddings_multiples=3,
                 generator=generator, width=width, height=height, num_inference_steps=30).images[0]
    
    image.save(f"{CACHE_DIR}/output/08_civitai_lora.png")

if __name__ == "__main__":
    with stub.run():
        run_lora.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/08_civitai_lora.png output', shell=True)
