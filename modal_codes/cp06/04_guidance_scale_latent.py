import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from transformers import AutoProcessor, CLIPVisionModelWithProjection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import modal
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "transformers", "scikit-learn", "matplotlib"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def generated_image_clip_embedding(width=256, height=256):
    device = "cuda"
    dtype = torch.float16 if "cuda" in device else torch.float32    
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=dtype, cache_dir=CACHE_DIR)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    clip_model = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch32", cache_dir=CACHE_DIR)
    clip_processor = AutoProcessor.from_pretrained(
        "openai/clip-vit-base-patch32", cache_dir=CACHE_DIR)

    n_images_per_prompt = 50
    result = []
    for gs in [1.5, 3, 7, 10, 20, 50]:
        generator = torch.Generator().manual_seed(1234)
        prompts = "a girl standing on the beach, smiling and having fun, blue haired, 1girl, border skirt, denim jacket, best quality, extremely detailed"
        negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
        images = pipe(prompt=prompts, negative_prompt=negative_prompt, generator=generator,
                      num_inference_steps=30, output_type="pil", width=width, height=height,
                      guidance_scale=gs, num_images_per_prompt=n_images_per_prompt).images # 100

        with torch.no_grad():
            clip_inputs = clip_processor(images=images, return_tensors="pt")
            outputs = clip_model(**clip_inputs)
            clip_latents = outputs.image_embeds.cpu().float().numpy()
            item = {"guidance_scale": gs, "clip_latents": clip_latents}
        result.append(item)

        torch.cuda.empty_cache()

    embeddings, guidance_scales = [], []
    for i, item in enumerate(result):
        x = item["clip_latents"]
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        embeddings.append(x)
        sim = x @ x.T
        guidance_scales.append(item["guidance_scale"])
        print(item["guidance_scale"], sim.mean(), np.std(x))
    embeddings = np.concatenate(embeddings)

    tsne = TSNE(n_components=2)
    embeddings_norm = tsne.fit_transform(embeddings)
    colors = list(mcolors.BASE_COLORS.keys())
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    for i in range(embeddings_norm.shape[0]//n_images_per_prompt):
        ax.scatter(embeddings_norm[i*n_images_per_prompt:(i+1)*n_images_per_prompt,0], embeddings_norm[i*n_images_per_prompt:(i+1)*n_images_per_prompt,1], color=colors[i], label=f"GS = {guidance_scales[i]}")
    ax.set_title("CLIP Image Embedding TSNE")
    ax.legend()
    fig.savefig(f"{CACHE_DIR}/output/04_guidance_scale_latent.png")

if __name__ == "__main__":
    with stub.run():
        generated_image_clip_embedding.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/04_guidance_scale_latent.png output', shell=True)
