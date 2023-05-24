from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch

def run_stable_diffusion(prompt,
                         output_filename, 
                         model_name="stabilityai/stable-diffusion-2-1-base",
                         device="cuda", seed=1234):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_tiling()
    pipe.to(device)

    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    generator = torch.Generator(device).manual_seed(seed)
    image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=30, generator=generator).images[0]
    image.save(output_filename)

if __name__ == "__main__":
    run_stable_diffusion("Cat in outer space, look at viewer, best quality",
                         "astro_cat.png", seed=1234)