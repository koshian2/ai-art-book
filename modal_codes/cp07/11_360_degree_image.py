import torch
from diffusers import UniPCMultistepScheduler, StableDiffusionPipeline
from utils import load_safetensors_lora, LORA_PATH
import modal
import subprocess
import requests
import os

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

def download_lora():
    urlData = requests.get("https://civitai.com/api/download/models/41292").content
    os.makedirs(LORA_PATH, exist_ok=True)
    with open(f"{LORA_PATH}/LatentLabs360.safetensors" ,mode='wb') as fp:
        fp.write(urlData)

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "transformers"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def render_image():
    download_lora()

    device = "cuda"
    model_id = "NoCrypt/SomethingV2_2"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe.enable_vae_tiling()
    pipe = load_safetensors_lora(
        pipe, f"{LORA_PATH}/LatentLabs360.safetensors", alpha=1.0
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device, torch.float16)

    generator = torch.Generator().manual_seed(1234)
    prompt = "a 360 equirectangular panorama, modelshoot style, bedroom in a rich mansion, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, 
                 generator=generator, width=1024, height=512,
                 num_inference_steps=30).images[0]
    image.save(f"{CACHE_DIR}/output/11_360dregree.png")

def main():
    # Modalでレンダリング
    with stub.run():
        render_image.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/11_360dregree.png output', shell=True)
    from omnidirectional_viewer import omniview

    # ローカルで実行
    # ESCボタンで終了
    omniview("output/11_360dregree.png", width=640, height=360)

if __name__ == "__main__":
    main()
