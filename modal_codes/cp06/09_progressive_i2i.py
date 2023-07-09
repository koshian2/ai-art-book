from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
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
def progressive_i2i(width=1920, height=1024):
    device = "cuda"    
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    prompt = "a girl adventurer is walking alone on a hill by the sea, mountains can be seen in the distance in the background, a small field of flowers, 1girl, blue shirts, green vest, knee-length skirt, blonded-hair, leather hair bands, masterpiece, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, " \
                    "fewer digits, cropped, worst quality, low quality"

    generator = torch.Generator().manual_seed(1234)
    # latent value
    randn = torch.randn((1, 4, height//16, width//16), generator=generator).to(device, torch.float16)
    # initial run
    initial_image = pipe(prompt=prompt, negative_prompt=negative_prompt, 
                         num_inference_steps=50, output_type="pil", latents=randn, guidance_scale=12).images[0]
    
    # upsampling
    upsampled_image = initial_image.resize((initial_image.width*2, initial_image.height*2), Image.Resampling.BICUBIC)

    pipe = StableDiffusionImg2ImgPipeline(**pipe.components)
    second_image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=upsampled_image,
                         num_inference_steps=30, output_type="pil", strength=0.7).images[0]
    second_image.save(f"{CACHE_DIR}/output/09_progressive_i2i_upsample.png")

    fig = plt.figure(figsize=(18, 7))
    for i, img in enumerate([initial_image, second_image]):
        ax = fig.add_subplot(1, 2, i+1)
        ax.imshow(img)
        ax.set_title("initial image" if i == 0 else "second image")
        ax.axis("off")
    fig.savefig(f"{CACHE_DIR}/output/09_progressive_i2i_compare.png")

if __name__ == "__main__":
    with stub.run():
        progressive_i2i.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/09_progressive_i2i* .', shell=True)
