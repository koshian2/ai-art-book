from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch
import matplotlib.pyplot as plt

def two_people_wo_latent_couple(width=960, height=512):
    device = "cuda:1"    
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    prompt = "two girls standing in a lavender field in the countryside and having fun, "
    prompt += "a girl enjoying the scent of flowers, beautiful girl with long blonde hair like a fairy tale princess, blue eyes, white dress, sandals, "
    prompt += "a girl taking a photo in a lavender field, healthy girl, wheat-colored skin tan, large eyes, colorful floral shirt, short cut hair, black hair, denim shorts"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
 
    # 乱数は1個で初期化
    generator = torch.Generator().manual_seed(1234)
    latent = torch.randn((1, 4, height//8, width//8), generator=generator).to(device, torch.float16)
    image = pipe(prompt=prompt, negative_prompt=negative_prompt,
                 latents=latent, num_inference_steps=50).images[0]
    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    two_people_wo_latent_couple()