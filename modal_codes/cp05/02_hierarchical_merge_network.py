import copy
import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import numpy as np
import torchvision
import modal
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

def hierarchal_merge_network(pipe_source, pipe_merge, attr, default_ratio, additional_ratio):
    merge_net = copy.deepcopy(getattr(pipe_source, attr))
    pipe_source_params = dict(getattr(pipe_source, attr).named_parameters())
    pipe_merge_params = dict(getattr(pipe_merge, attr).named_parameters())
    for key, param in merge_net.named_parameters():
        ratio = default_ratio
        for layer_name, layer_specific_ratio in additional_ratio.items():
            if key.startswith(layer_name):
                ratio = layer_specific_ratio
                break
        x = pipe_source_params[key] * (1-ratio) + pipe_merge_params[key] * ratio
        param.data = x
    return merge_net

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "torchvision", "transformers"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def hierarchal_merge_compare():
    device = "cuda"
    pipe_pastel = StableDiffusionPipeline.from_pretrained(
        "JamesFlare/pastel-mix", torch_dtype=torch.float16,
        cache_dir=CACHE_DIR, safety_checker=None)    

    prompt = "1girl, silver hair, full body, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, extra legs, cropped, worst quality, low quality"

    merge_ratios = [0, 0.25, 0.5, 0.75, 1]

    unet_merge_template = [
        lambda r: {"down_blocks.0": r, "up_blocks.3": r},
        lambda r: {"mid_block": r}
    ]
    for i, template_func in enumerate(unet_merge_template):
        displayed_images = []
        for merge_ratio in merge_ratios:
            pipe = StableDiffusionPipeline.from_pretrained(
                "prompthero/openjourney-v4", torch_dtype=torch.float16, 
                cache_dir=CACHE_DIR, safety_checker=None)
            specific_merge_config = template_func(merge_ratio)
            pipe.unet = hierarchal_merge_network(pipe, pipe_pastel, "unet", 0.0, specific_merge_config) # merge unet only
            pipe.text_encoder = hierarchal_merge_network(pipe, pipe_pastel, "text_encoder", 0.0, {})
            pipe.enable_vae_tiling()
            pipe.to(device)

            generator = torch.Generator().manual_seed(1234)
            images_merge = pipe(prompt=prompt, negative_prompt=negative_prompt, generator=generator, 
                                num_images_per_prompt=3, num_inference_steps=30).images
            displayed_images.append(images_merge)
        displayed_images = sum(displayed_images, start=[])
        displayed_images = np.array([np.array(x) for x in displayed_images]) / 255.0
        displayed_images = torch.from_numpy(displayed_images).permute(0, 3, 1, 2)
        torchvision.utils.save_image(displayed_images, f"{CACHE_DIR}/output/02_hierarchal_merge_{i}.png", nrow=3)

if __name__ == "__main__":
    with stub.run():
        hierarchal_merge_compare.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/02_hierarchal_merge* .', shell=True)
