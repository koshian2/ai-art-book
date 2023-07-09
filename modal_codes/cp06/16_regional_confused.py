from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch
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
def two_people_wo_latent_couple(width=960, height=512):
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    prompt = "two girls standing in a lavender field in the countryside and having fun, "
    prompt += "a girl enjoying the scent of flowers, beautiful girl with long blonde hair like a fairy tale princess, blue eyes, white dress, sandals, "
    prompt += "a girl taking a photo in a lavender field, healthy girl, wheat-colored skin tan, large eyes, colorful floral shirt, short cut hair, black hair, denim shorts"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
 
    # 乱数は1個で初期化
    generator = torch.Generator().manual_seed(1234)
    latent = torch.randn((1, 4, height//8, width//8), generator=generator).to(device, torch.float16)
    image = pipe(prompt=prompt, negative_prompt=negative_prompt,
                 latents=latent, num_inference_steps=50).images[0]
    image.save(f"{CACHE_DIR}/output/16_regional_confused.png")

if __name__ == "__main__":
    with stub.run():
        two_people_wo_latent_couple.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/16_regional_confused.png output', shell=True)
