from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch
import matplotlib.pyplot as plt
from utils import LORA_PATH, load_safetensors_lora

def run_flat_lora(prompt,
                  model_name="stabilityai/stable-diffusion-2-1-base",
                  device="cuda", seed=1234):
    fig = plt.figure(figsize=(18, 10))
    for i, lora_name in enumerate(["bigeye.safetensors", "flat2.safetensors"]):
        alphas = [-1, -0.5, 0, 0.5, 1]
        for j, alpha in enumerate(alphas):
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name, torch_dtype=torch.float16)
            pipe = load_safetensors_lora(pipe, f"{LORA_PATH}/{lora_name}", alpha=alpha)

            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.safety_checker=lambda images, **kwargs: (images, False)
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

    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    run_flat_lora("1girl, look at viewer, best quality", model_name="H:/diffusion_models/diffusers/AnythingV5_v5PrtRE")
