import copy
import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import numpy as np
import torchvision

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

def hierarchal_merge_compare():
    device = "cuda:1"
    pipe_pastel = StableDiffusionPipeline.from_pretrained(
        "andite/pastel-mix", torch_dtype=torch.float16)    

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
                "prompthero/openjourney-v4", torch_dtype=torch.float16)        
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            specific_merge_config = template_func(merge_ratio)
            pipe.unet = hierarchal_merge_network(pipe, pipe_pastel, "unet", 0.0, specific_merge_config) # merge unet only
            pipe.text_encoder = hierarchal_merge_network(pipe, pipe_pastel, "text_encoder", 0.0, {})
            pipe.safety_checker = lambda images, **kwargs: (images, False)
            pipe.enable_vae_tiling()
            pipe.to(device)

            generator = torch.Generator().manual_seed(1234)
            images_merge = pipe(prompt=prompt, negative_prompt=negative_prompt, generator=generator, 
                                num_images_per_prompt=3, num_inference_steps=30).images
            displayed_images.append(images_merge)
        displayed_images = sum(displayed_images, start=[])
        displayed_images = np.array([np.array(x) for x in displayed_images]) / 255.0
        displayed_images = torch.from_numpy(displayed_images).permute(0, 3, 1, 2)
        torchvision.utils.save_image(displayed_images, f"output/06/10_hierarchal_merge_{i}.png", nrow=3)

if __name__ == "__main__":
    hierarchal_merge_compare()
