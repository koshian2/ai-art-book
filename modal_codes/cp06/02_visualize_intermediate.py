import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import numpy as np
import matplotlib.pyplot as plt
import modal
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

class IntermediateCallback:
    def __init__(self, pipe):
        self.pipe = pipe
        self.steps = []
        self.latents = []
        self.decode_images = []

    def callback(self, i, t, latent):
        print(i)
        self.steps.append(i)
        l_array = latent.detach().cpu().numpy()
        self.latents.append(l_array)
        pic = (self.pipe.decode_latents(latent)[0]*255.0).astype(np.uint8)
        self.decode_images.append(pic)

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "transformers", "matplotlib"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def visualize_intermediate(width=960, height=512):
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    generator = torch.Generator().manual_seed(1234)
    prompts = "a girl standing on the beach, smiling and having fun, blue haired, 1girl, border skirt, denim jacket, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    cb = IntermediateCallback(pipe)
    image = pipe(prompt=prompts, negative_prompt=negative_prompt, generator=generator,
                 num_inference_steps=50, output_type="pil", width=width, height=height,
                 callback=cb.callback, callback_steps=5).images[0]
    result = []
    for i, latent, pic in zip(cb.steps, cb.latents, cb.decode_images):
        item = {
            "step": i,
            "latent": latent,
            "pic": pic
        }
        result.append(item)

    fig = plt.figure(figsize=(15, 5))
    for i, step_idx in enumerate([0, 3, 6, 9]):
        ax = fig.add_subplot(2, 4, 2*i+1)
        latent = result[step_idx]["latent"][0, :3].transpose(1, 2, 0).astype(np.float32)
        latent = (latent-latent.min())/(latent.max()-latent.min())
        latent = (latent*255.0).astype(np.uint8)

        ax.imshow(latent)
        ax.set_title(f"latent t={result[step_idx]['step']}")
        ax.set_axis_off()

        ax = fig.add_subplot(2, 4, 2*i+2)
        ax.imshow(result[step_idx]["pic"])
        ax.set_title(f"decode t={result[step_idx]['step']}")
        ax.set_axis_off()
    fig.savefig(f"{CACHE_DIR}/output/02_visualize_intermediate.png")

if __name__ == "__main__":
    with stub.run():
        visualize_intermediate.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/02_visualize_intermediate.png output', shell=True)
