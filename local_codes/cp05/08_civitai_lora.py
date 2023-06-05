import torch
from diffusers import DiffusionPipeline, UniPCMultistepScheduler
import matplotlib.pyplot as plt
from utils import load_safetensors_lora
from settings import LORA_DIRECTORY

# settings.pyにLoRAを保存したディレクトリを記述

def run_lora(width=512, height=960):
    device = "cuda"    
    pipe = DiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16,
         custom_pipeline="lpw_stable_diffusion")
    pipe = load_safetensors_lora(pipe, f"{LORA_DIRECTORY}/lumine1-000008.safetensors", alpha=0.5)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    prompt = "luminedef, luminernd, 1girl, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    generator = torch.Generator(device).manual_seed(1234)
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, max_embeddings_multiples=3,
                 generator=generator, width=width, height=height, num_inference_steps=30).images[0]

    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    run_lora()
