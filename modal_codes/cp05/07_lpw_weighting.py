from diffusers import DiffusionPipeline, UniPCMultistepScheduler
import torch
import matplotlib.pyplot as plt
from settings import MODEL_DIRECTORY
import modal
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "transformers", "matplotlib"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def run_lpw_weighting():
    device = "cuda"
    pipe = DiffusionPipeline.from_pretrained(
        f"{MODEL_DIRECTORY}/merge_Counterfeit-V3.0_orangemix", torch_dtype=torch.float16,
         custom_pipeline="lpw_stable_diffusion"
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = lambda images, **kwargs: (images, False)
    pipe.enable_vae_tiling()
    pipe.to(device)

    fig = plt.figure(figsize=(18, 8))
    for i, weight in enumerate([0.7, 0.8, 1, 1.3, 1.6]):
        prompt = f"1girl, look at viewer, school girl, (beige dress:{weight}), masterpiece, best quality"
        negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

        generator = torch.Generator(device).manual_seed(1234)
        images = pipe.text2img(prompt, negative_prompt=negative_prompt, generator=generator, num_inference_steps=30,
                            width=512, height=960).images[0]
        ax = fig.add_subplot(1, 5, i+1)
        ax.set_title(f"weight={weight}")
        ax.imshow(images)

    fig.savefig(f"{CACHE_DIR}/output/07_lpw_weighting.png")
    
if __name__ == "__main__":
    with stub.run():
        run_lpw_weighting.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/07_lpw_weighting.png output', shell=True)
