import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

def noise_scale(width=960, height=512):
    device = "cuda:1"
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    result = []
    # Guidance Scale by specifing latent
    prompts = "great panorama of snowy mountains, best quality, extremely detailed"
    negative_prompt = "worst quality, low quality"
    for gs in [1.5, 7, 10, 20, 50, 100]:
        generator = torch.Generator().manual_seed(1234)
        # Latentを直接指定してもいける
        latent = torch.randn((1, 4, height//8, width//8), generator=generator).to(device, torch.float16)
        image = pipe(prompt=prompts, negative_prompt=negative_prompt, guidance_scale=gs,
                    num_inference_steps=50, output_type="pil", latents=latent).images[0]
        result.append({"mode": "Guidance Scale", "value": gs, "pic": image})

    for scale in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        generator = torch.Generator().manual_seed(1234)
        # Noise scale
        latent = torch.randn((1, 4, height//8, width//8), generator=generator).to(device, torch.float16) * scale
        image = pipe(prompt=prompts, negative_prompt=negative_prompt,
                    num_inference_steps=50, output_type="pil", latents=latent).images[0]
        result.append({"mode": "Noise Scale", "value": scale, "pic": image})

    with open("output/05/06_noise_scale.pkl", "wb") as fp:
        pickle.dump(result, fp)

import pickle
import matplotlib.pyplot as plt

def visualize():
    with open("output/05/06_noise_scale.pkl", "rb") as fp:
        data = pickle.load(fp)
    fig = plt.figure(figsize=(16, 12))
    for i, item in enumerate(data):
        ax = fig.add_subplot(4, 3, i+1)
        ax.imshow(item["pic"])
        ax.axis("off")
        ax.set_title(f"{item['mode']}={item['value']}")
    fig.savefig("output/05/07_noise_scale.png")
    plt.show()

if __name__ == "__main__":
    # noise_scale()
    visualize()
