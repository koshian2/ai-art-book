from diffusers import DiffusionPipeline, UniPCMultistepScheduler
import torch
import modal
import subprocess
import os

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "transformers", "omegaconf"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def run_lpw():
    prompt = "On a quiet cobblestone street in a quaint old town, a little girl around 5-6 years old is happily splashing through puddles in the pouring rain during a cloudy, late afternoon just as the sun is setting. She is wearing a yellow raincoat with a hood, yellow boots, and carrying a small backpack. In her hand, she holds a pink umbrella with a frog pattern, about to open it. The scene is filled with warm colors of yellow, pink, green, and brown, expressing the delightful atmosphere despite the rain. The street is lined with trees, red postal box, and old wooden houses, with the sound of raindrops falling from the branches. Wet fallen autumn leaves are scattered about, adding to the season's ambiance. The old streetlamp with its vintage design is not lit yet, casting a gentle darkness around the girl as she continues on her merry way."
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    device = "cuda"
    pipe = DiffusionPipeline.from_pretrained(
        f"{CACHE_DIR}/merge_Counterfeit-V3.0_orangemix", torch_dtype=torch.float16,
         custom_pipeline="lpw_stable_diffusion"
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = lambda images, **kwargs: (images, False)
    pipe.enable_vae_tiling()
    pipe.to(device)

    generator = torch.Generator(device).manual_seed(1234)
    images = pipe.text2img(prompt, negative_prompt=negative_prompt, generator=generator, num_inference_steps=30,
                           width=960, height=512, max_embeddings_multiples=10).images[0]
    
    output_filename = f"{CACHE_DIR}/output/05_run_lpw_diffusion.png"
    os.makedirs(os.path.basename(output_filename), exist_ok=True)
    images.save(output_filename)
        
if __name__ == "__main__":
    with stub.run():
        run_lpw.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/05_run_lpw_diffusion.png output', shell=True)
