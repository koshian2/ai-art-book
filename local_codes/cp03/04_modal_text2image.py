from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch
import modal
import io

CACHE_DIR = "/cache"
volume = modal.SharedVolume().persist("model-cache-vol")

stub = modal.Stub()

@stub.function(
    image=modal.Image.debian_slim().pip_install("torch", "diffusers[torch]", "transformers", "accelerate"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    shared_volumes={CACHE_DIR: volume}
)
def run_stable_diffusion(prompt,
                         model_name,
                         seed):
    device="cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_tiling()
    pipe.to(device)

    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    generator = torch.Generator(device).manual_seed(seed)
    image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=30, generator=generator).images[0]

    with io.BytesIO() as buf:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        return img_bytes        

def main(prompt, output_filename, 
         model_name="stabilityai/stable-diffusion-2-1-base", 
         seed=1234):
    with stub.run():
        img_bytes = run_stable_diffusion.call(prompt, model_name, seed)
        with open(output_filename, "wb") as fp:
            fp.write(img_bytes)    

if __name__ == "__main__":
    main("Cat in outer space, look at viewer, best quality",
          "astro_cat.png", seed=1235)