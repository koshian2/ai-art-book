import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import numpy as np
from transformers import pipeline
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
def noise_depth_mask(width=960, height=512):
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    prompts = "masterpiece, best quality, 1girl, brown-haired girl, autumn leaves, standing, beauty, harmony, nature, breathtaking, foliage, dancing, picturesque, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    generator = torch.Generator().manual_seed(1234)
    # latent value
    randn = torch.randn((1, 4, height//8, width//8), generator=generator).to(device, torch.float16)
    # initial run
    initial_image = pipe(prompt=prompts, negative_prompt=negative_prompt, 
                         num_inference_steps=50, output_type="pil", latents=randn).images[0]
    
    depth_estimator = pipeline('depth-estimation', cache_dir=CACHE_DIR)
    depth_map = depth_estimator(initial_image)['depth']
    depth_map = depth_map.resize((width//8, height//8), Image.Resampling.BOX)
    depth_mask = np.array(depth_map) / 255.0 * 0.5 + 0.8
    depth_mask = np.stack([depth_mask for i in range(4)], axis=0)[None]
    masked_randn = randn * torch.from_numpy(depth_mask).to(device, torch.float16)
    # second run
    second_image = pipe(prompt=prompts, negative_prompt=negative_prompt, 
                        num_inference_steps=50, output_type="pil", latents=masked_randn).images[0]
    
    fig = plt.figure(figsize=(18, 7))
    for i, img in enumerate([initial_image, second_image]):
        ax = fig.add_subplot(1, 2, i+1)
        ax.imshow(img)
        ax.set_title(("w/o" if i == 0 else "with") + " depth mask")
        ax.axis("off")
    fig.savefig(f"{CACHE_DIR}/output/07_noise_depth_mask.png")

if __name__ == "__main__":
    with stub.run():
        noise_depth_mask.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/07_noise_depth_mask.png output', shell=True)
    