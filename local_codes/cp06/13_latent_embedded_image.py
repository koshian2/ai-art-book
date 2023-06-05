import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import matplotlib.pyplot as plt

def embedded_latent(second_prompts, width=1920, height=1024):
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)
    prompt = "a girl adventurer is walking alone on a hill by the sea, mountains can be seen in the distance in the background, a small field of flowers, 1girl, blue shirts, green vest, knee-length skirt, blonded-hair, leather hair bands, masterpiece, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    result = []
    with torch.no_grad():
        # 1回目の出力
        generator = torch.Generator().manual_seed(1234)
        noise_large = torch.randn((1,4,height//8,width//8), generator=generator).to(device, torch.float16)
        noise_small = noise_large[:, :, ::2, ::2]
        latent_small = pipe(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=12,
                            latents=noise_small, num_inference_steps=50, output_type="latent").images
        image_small = pipe.numpy_to_pil(pipe.decode_latents(latent_small))[0]
        result.append(image_small)

        # 2回目の出力
        x = noise_large.clone()
        x[:, :, ::2, ::2] = latent_small * 0.8 + noise_small * 0.2
        for prompt in second_prompts:
            image_big = pipe(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=12,
                            latents=x, num_inference_steps=50, output_type="pil").images[0]
            result.append(image_big)
        return result

def main():
    second_prompts = [
        "a summer scene, masterpiece, best quality, extremely detailed",
        "an autumn scene, masterpiece, best quality, extremely detailed",
        "a winter scene, masterpiece, best quality, extremely detailed",
    ]
    results = embedded_latent(second_prompts)
    fig = plt.figure(figsize=(20, 12))
    titles = ["base_image", "a summer scene", "an autumn scene", "a winter scene"]
    for i, img in enumerate(results):
        ax = fig.add_subplot(2, 2, i+1)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(titles[i])
    plt.show()

if __name__ == "__main__":
    main()