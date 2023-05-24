from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler, StableDiffusionImg2ImgPipeline
import torch
import torch.nn.functional as F

def failed_latent_upsampling(width=1920, height=1024):
    device = "cuda"    
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    prompt = "a girl adventurer is walking alone on a hill by the sea, mountains can be seen in the distance in the background, a small field of flowers, 1girl, blue shirts, green vest, knee-length skirt, blonded-hair, leather hair bands, masterpiece, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, " \
                    "fewer digits, cropped, worst quality, low quality"

    results, titles = [], []
    # initial latent
    generator = torch.Generator().manual_seed(1234)
    randn = torch.randn((1, 4, height//16, width//16), generator=generator).to(device, torch.float16)
    # vae latent
    vae_latent = pipe(prompt=prompt, negative_prompt=negative_prompt, 
                      num_inference_steps=50, output_type="latent", latents=randn, guidance_scale=12).images
    for method in ["nearest", "bicubic"]:
        with torch.no_grad():
            x = F.interpolate(vae_latent, scale_factor=2, mode=method)
            image = pipe.numpy_to_pil(pipe.decode_latents(x))
            results.append(image[0])
            titles.append("VAE Latent x2 - "+method)

    for method in ["nearest", "bicubic"]:
        with torch.no_grad():
            x = F.interpolate(randn, scale_factor=2, mode=method)
        image = pipe(prompt=prompt, negative_prompt=negative_prompt,
                     num_inference_steps=50, latents=x, guidance_scale=12).images
        results.append(image[0])
        titles.append("Initial Latent x2 - "+method)
