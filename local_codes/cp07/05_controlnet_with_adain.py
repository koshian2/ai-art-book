
import torch
from safetensors.torch import load_file

from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from stable_diffusion_reference import StableDiffusionReferencePipeline
from PIL import Image
import matplotlib.pyplot as plt

def load_resize(file_name):
    image = Image.open(file_name)
    w, h = image.size
    w, h = (x - x % 8 for x in (w, h))
    image = image.resize((w, h), resample=Image.LANCZOS)    
    return image

def main():
    initial_img = Image.open("data/black_hair_girl.png")

    device = "cuda:1"
    pipe = StableDiffusionReferencePipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_tiling()
    pipe.to(device, torch.float16)

    prompt = "best quality"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    fig = plt.figure(figsize=(12, 12))
    for fi, fidelity in enumerate([0, 0.5, 1]):
        generator = torch.Generator().manual_seed(1234)
        image = pipe(prompt=prompt, ref_image=initial_img, negative_prompt=negative_prompt, style_fidelity=fidelity, reference_adain=True,
                    generator=generator, num_inference_steps=30, num_images_per_prompt=5).images

        for i in range(5):
            ax = fig.add_subplot(3, 5, fi*5+i+1)
            ax.imshow(image[i])
            ax.axis("off")
            if i == 0:
                ax.set_title(f"style_fidelity={fidelity}")
    fig.savefig("output/07/03_reference_with_adain.png")
    plt.show()

if __name__ == "__main__":
    main()
