from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch
import matplotlib.pyplot as plt

def latent_visual_prompt_inpainting(width=1920, height=1024):
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)
    prompt = "a girl adventurer is walking alone on a hill by the sea, mountains can be seen in the distance in the background, a small field of flowers, 1girl, blue shirts, green vest, knee-length skirt, blonded-hair, leather hair bands, masterpiece, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    with torch.no_grad():
        generator = torch.Generator().manual_seed(1234)
        noise_large = torch.randn((1,4,height//8,width//8), generator=generator).to(device, torch.float16)
        latent_first = pipe(prompt=prompt, negative_prompt=negative_prompt, 
                         num_inference_steps=50, output_type="latent", latents=noise_large, guidance_scale=12).images
        image_first = pipe.numpy_to_pil(pipe.decode_latents(latent_first))[0]

        # Second
        x = noise_large.clone()
        _, _, h, w = latent_first.shape
        # latent visual promptを左1/3にかける
        x[:, :, ::2, :w//3:2] = latent_first[:, :, ::2, :w//3:2] * 0.8 + x[:, :, ::2, :w//3:2] * 0.2
        # 人が分裂しないように、高さ1/3～2/3、右2/3の温度を下げる
        x[:, :, h//3:h*2//3, w//3:] *= 0.9
        latent_second = pipe(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=12,
                            latents=x, num_inference_steps=50, output_type="latent").images
        image_second = pipe.numpy_to_pil(pipe.decode_latents(latent_second))[0]

    fig = plt.figure(figsize=(18, 7))
    for i, img in enumerate([image_first, image_second]):
        ax = fig.add_subplot(1, 2, i+1)
        ax.imshow(img)
        ax.set_title("initial image" if i == 0 else "after temperature adjusting")
        ax.axis("off")
    plt.show()

if __name__ == "__main__":
    latent_visual_prompt_inpainting()