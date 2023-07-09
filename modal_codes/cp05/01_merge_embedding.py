import copy
import torch
import pickle
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from transformers import CLIPVisionModelWithProjection, AutoProcessor
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import torchvision
import modal
import subprocess

### --- Important ---
### CP04の05_cache_embedding.pyを先に実行して、「embedding_20k.pkl」を「data」フォルダに配置してください
### -----------------

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

def merge_network(pipe_source, pipe_merge, attr, ratio):
    merge_net = copy.deepcopy(getattr(pipe_source, attr))
    pipe_source_params = dict(getattr(pipe_source, attr).named_parameters())
    pipe_merge_params = dict(getattr(pipe_merge, attr).named_parameters())
    for key, param in merge_net.named_parameters():
        x = pipe_source_params[key] * (1-ratio) + pipe_merge_params[key] * ratio
        param.data = x
    return merge_net

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "torchvision", "transformers", "matplotlib", "scikit-learn"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    timeout=1200,
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def main():
    device = "cuda"
    pipe_pastel = StableDiffusionPipeline.from_pretrained(
        "JamesFlare/pastel-mix", torch_dtype=torch.float16, cache_dir=CACHE_DIR)    

    prompt = "1girl, silver hair, full body, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, extra legs, cropped, worst quality, low quality"

    result = []
    displayed_images = []
    merge_ratios = [0, 0.25, 0.5, 0.75, 1]
    # merge_ratios = [0, 1]
    n_networks = len(merge_ratios)
    for merge_ratio in merge_ratios:
        pipe = StableDiffusionPipeline.from_pretrained(
            "prompthero/openjourney-v4", torch_dtype=torch.float16, cache_dir=CACHE_DIR, safety_checker=None)        
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.unet = merge_network(pipe, pipe_pastel, "unet", merge_ratio)
        pipe.text_encoder = merge_network(pipe, pipe_pastel, "text_encoder", merge_ratio)
        pipe.enable_vae_tiling()
        pipe.to(device)

        generator = torch.Generator().manual_seed(1234)
        for i in range(5):
            images_merge = pipe(prompt=prompt, negative_prompt=negative_prompt, generator=generator, 
                                num_images_per_prompt=10, num_inference_steps=30).images
            result.append(images_merge)
        displayed_images.append(images_merge[:4])
    result = sum(result, start=[])
    displayed_images = sum(displayed_images, start=[])
    displayed_images = np.array([np.array(x) for x in displayed_images]) / 255.0
    displayed_images = torch.from_numpy(displayed_images).permute(0, 3, 1, 2)

    output_filename = f"{CACHE_DIR}/output/01_merge_networks.jpg"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    torchvision.utils.save_image(displayed_images, output_filename, quality=92, nrow=4)

    clip_model = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-large-patch14", cache_dir=CACHE_DIR).to(device)
    clip_processor = AutoProcessor.from_pretrained(
        "openai/clip-vit-large-patch14", cache_dir=CACHE_DIR)

    with open("data/embedding_20k.pkl", "rb") as fp:
        data = pickle.load(fp)

    with torch.no_grad():
        inputs = clip_processor(images=result, return_tensors="pt").to(device)
        outputs = clip_model(**inputs)
        x = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        x = x.cpu().numpy()
    
    tsne = TSNE(n_components=2, perplexity=10)
    normed_x = tsne.fit_transform(x)
    colors = list(mcolors.TABLEAU_COLORS.keys())

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    for i in range(n_networks):
        n = normed_x.shape[0] // n_networks
        y = normed_x[i*n:(i+1)*n]
        ax.scatter(y[:, 0], y[:, 1], c=colors[i], label=f"jorney={(1-merge_ratios[i])}, pastel={merge_ratios[i]}")
        if i > 0:
            image_diff = x[i*n:(i+1)*n] - x[:n]
            image_diff /= np.linalg.norm(image_diff, axis=-1, keepdims=True)
            similarity = (image_diff @ data["embedding"].T)[0]

            max_indices = np.argsort(similarity)[::-1][:10]
            print("\n---", merge_ratios[i], "---")
            print(np.array(data["word"])[max_indices].tolist())
            print((similarity[max_indices]*100).astype(np.int32).tolist())

    fig.legend()
    
    output_filename = f"{CACHE_DIR}/output/01_merge_embedding.png"
    fig.savefig(output_filename)

if __name__ == "__main__":
    with stub.run():
        main.call()
    os.makedirs("output", exist_ok=True)
    subprocess.run(
        f'modal nfs get model-cache-vol output/01_merge_* .', shell=True)
