import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from transformers import CLIPTextModel
import matplotlib.pyplot as plt
import modal
import os
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

def get_indices(pipe, prompt: str):
    """Utility function to list the indices of the tokens you wish to alte"""
    ids = pipe.tokenizer(prompt).input_ids
    indices = {i: tok for tok, i in zip(pipe.tokenizer.convert_ids_to_tokens(ids), range(len(ids)))}
    return indices

@stub.function(
    image=modal.Image.debian_slim().pip_install("torch", "diffusers[torch]", "transformers", "matplotlib"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def main():
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        "JosephusCheung/ACertainModel", torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_tiling()
    pipe.to(device)

    prompt = "Smiling girl with Christmas tree in background, green hair, knit sweater, night view, best quality"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    print(get_indices(pipe, prompt))

    images, norms = [], []
    for i in range(8):
        pipe.text_encoder = CLIPTextModel.from_pretrained("JosephusCheung/ACertainModel", 
                                                          subfolder="text_encoder", 
                                                          num_hidden_layers=11-i, torch_dtype=torch.float16,
                                                          cache_dir=CACHE_DIR).to(device)
        generator = torch.Generator(device).manual_seed(1234)
        prompt_embed = pipe._encode_prompt(prompt, device, negative_prompt=negative_prompt, num_images_per_prompt=1, do_classifier_free_guidance=True)
        image = pipe(prompt_embeds=prompt_embed[1:2], negative_prompt_embeds=prompt_embed[0:1], generator=generator, num_inference_steps=30,
                     width=960).images[0]
        images.append(image)
        norm = prompt_embed[1].std(dim=-1)
        norms.append(norm)

    os.makedirs(f"{CACHE_DIR}/output", exist_ok=True)
    fig = plt.figure(figsize=(18, 10))
    for i in range(len(images)):
        ax = fig.add_subplot(2, 4, i+1)
        ax.imshow(images[i])
        ax.set_title(f"Clip skip = {i+1}")
        ax.axis("off")
    fig.savefig(f"{CACHE_DIR}/output/15_clip_skip_compare_1.png")
    fig = plt.figure(figsize=(18, 8))
    for i in range(len(norms)):
        ax = fig.add_subplot(2, 4, i+1)
        ax.plot(norms[i][:20].detach().cpu().numpy())
        ax.set_title(f"Clip skip = {i+1}")
        ax.set_ylim((1.0, 1.2))
    fig.savefig(f"{CACHE_DIR}/output/15_clip_skip_compare_2.png")

if __name__ == "__main__":
    with stub.run():
        main.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/15_clip_skip_compare* .', shell=True)
