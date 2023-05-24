import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import numpy as np
import matplotlib.pyplot as plt

def noise_spatial_mask(width=960, height=512, mask_min=0.75, mask_max=1.25):
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    result = []
    masks = [
        # left -> right
        np.tile(np.linspace(mask_min, mask_max, width//8, dtype=np.float16)[None, None, None, :], (1, 4, height//8, 1)),
        # left <- right
        np.tile(np.linspace(mask_max, mask_min, width//8, dtype=np.float16)[None, None, None, :], (1, 4, height//8, 1)),
        # left -> middle <- right
        np.tile(
            np.concatenate([
                np.linspace(mask_min, mask_max, width//16, dtype=np.float16),
                np.linspace(mask_max, mask_min, width//16, dtype=np.float16)
            ])[None, None, None, :], (1, 4, height//8, 1)
        ),
        # top -> bottom
        np.tile(np.linspace(mask_min, mask_max, height//8, dtype=np.float16)[None, None, :, None], (1, 4, 1, width//8)),
    ]
    names = [
        f"left[{mask_min}] -> right[{mask_max}]", 
        f"left[{mask_max}] <- right[{mask_min}]", 
        f"left[{mask_min}] -> middle[{mask_max}] <- right[{mask_min}]", 
        f"top[{mask_min}] -> bottom[{mask_max}]"
    ]

    prompts = "great panorama of snowy mountains, best quality, extremely detailed"
    negative_prompt = "worst quality, low quality"
    for mask in masks:
        generator = torch.Generator().manual_seed(1234)
        # Noise scale
        latent = torch.randn((1, 4, height//8, width//8), generator=generator).to(device, torch.float16)
        mask_tensor = torch.from_numpy(mask).to(device, torch.float16)
        latent = latent * mask_tensor
        image = pipe(prompt=prompts, negative_prompt=negative_prompt,
                    num_inference_steps=50, output_type="pil", latents=latent).images[0]
        result.append(image)

    fig = plt.figure(figsize=(18, 10))
    for i in range(len(result)):
        ax = fig.add_subplot(2, 2, i+1)
        ax.imshow(result[i])
        ax.axis("off")
        ax.set_title(f"{names[i]}")
    plt.show()