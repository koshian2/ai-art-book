from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from transformers import CLIPVisionModelWithProjection, AutoProcessor, CLIPTextModel
import torch
import matplotlib.pyplot as plt
import numpy as np
import modal
import os
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

# 遅いのでインスタンスはa10gでもいいかも
@stub.function(
    image=modal.Image.debian_slim().pip_install("torch", "diffusers[torch]", "transformers", "matplotlib"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    timeout=1800,
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def clip_skip_image_variation():
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        "JosephusCheung/ACertainModel", torch_dtype=torch.float16, cache_dir=CACHE_DIR, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_tiling()
    pipe.to(device)

    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", cache_dir=CACHE_DIR)
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=CACHE_DIR)

    prompt = "Smiling girl with Christmas tree in background, green hair, knit sweater, night view, best quality"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    result = {}
    for i in range(8):
        image_embeddings = []
        pipe.text_encoder = CLIPTextModel.from_pretrained("JosephusCheung/ACertainModel", 
                                                        subfolder="text_encoder", 
                                                        num_hidden_layers=11-i, torch_dtype=torch.float16,
                                                        cache_dir=CACHE_DIR).to(device)
        for j in range(5):
            generator = torch.Generator(device).manual_seed(1234+j)
            image = pipe(prompt=prompt, negative_prompt=negative_prompt, generator=generator, num_inference_steps=30,
                        num_images_per_prompt=8).images[0]
            with torch.no_grad():
                inputs = clip_processor(images=image, return_tensors="pt")
                outputs = clip_model(**inputs) 
                image_embeddings.append((outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)).numpy())

        image_embeddings = np.concatenate(image_embeddings, axis=0)
        result[i] = image_embeddings.std(axis=-1).mean()

    values = np.array(list(result.values()))
    os.makedirs(f"{CACHE_DIR}/output", exist_ok=True)
    plt.bar(result.keys(), result.values())
    plt.ylim((values.min()*0.99, values.max()*1.01))
    plt.savefig(f"{CACHE_DIR}/output/16_clip_skip_image_variation.png")
    print(result)

if __name__ == "__main__":
    with stub.run():
        clip_skip_image_variation.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/16_clip_skip_image_variation* .', shell=True)
