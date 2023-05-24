
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
    initial_img = Image.open("data/texture_leaf.jpg")

    device = "cuda:1"
    pipe = StableDiffusionReferencePipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device, torch.float16)

    prompt = "A girl like a doll, silver hair, 1girl, look at viewer, face, blue eyes, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    fig = plt.figure(figsize=(18, 7))
    for fi, use_adain in enumerate([False, True]):
        generator = torch.Generator().manual_seed(1234)
        image = pipe(prompt=prompt, ref_image=initial_img, negative_prompt=negative_prompt, style_fidelity=0, reference_adain=use_adain,
                    generator=generator, num_inference_steps=30, num_images_per_prompt=4).images

        for i in range(4):
            ax = fig.add_subplot(2, 4, fi*4+i+1)
            ax.imshow(image[i])
            ax.axis("off")
            if i == 0:
                ax.set_title(f"AdaIN = {use_adain}")
    fig.savefig("output/07/03_reference_with_adain.png")
    plt.show()

if __name__ == "__main__":
    main()
