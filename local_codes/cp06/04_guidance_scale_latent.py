import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from transformers import AutoProcessor, CLIPVisionModelWithProjection
import io
import numpy as np

def generated_image_clip_embedding(width=256, height=256):
    device = "cuda"
    dtype = torch.float16 if "cuda" in device else torch.float32    
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    result = []
    for gs in [1.5, 3, 7, 10, 20, 50]:
        generator = torch.Generator().manual_seed(1234)
        prompts = "a girl standing on the beach, smiling and having fun, blue haired, 1girl, border skirt, denim jacket, best quality, extremely detailed"
        negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
        images = pipe(prompt=prompts, negative_prompt=negative_prompt, generator=generator,
                      num_inference_steps=30, output_type="pil", width=width, height=height,
                      guidance_scale=gs, num_images_per_prompt=100).images

        with torch.no_grad():
            clip_inputs = clip_processor(images=images, return_tensors="pt")
            outputs = clip_model(**clip_inputs)
            clip_latents = outputs.image_embeds.cpu().float().numpy()
            item = {"guidance_scale": gs, "clip_latents": clip_latents}
        result.append(item)

        torch.cuda.empty_cache()
    return result

import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors

def visualize():
    with open("output/05/04_2_intermediate.pkl", "rb") as fp:
        data = pickle.load(fp)
    embeddings, guidance_scales = [], []
    for i, item in enumerate(data):
        with io.BytesIO() as buf:
            buf.write(item["clip_latents"])
            buf.seek(0)
            x = np.load(buf)
            x /= np.linalg.norm(x, axis=-1, keepdims=True)
            embeddings.append(x)
            sim = x @ x.T
            guidance_scales.append(item["guidance_scale"])
            print(item["guidance_scale"], sim.mean(), np.std(x))
    embeddings = np.concatenate(embeddings)

    # tsne = TSNE(n_components=2)
    tsne = PCA(n_components=2)
    embeddings_norm = tsne.fit_transform(embeddings)
    colors = list(mcolors.BASE_COLORS.keys())
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    for i in range(embeddings_norm.shape[0]//100):
        ax.scatter(embeddings_norm[i*100:(i+1)*100,0], embeddings_norm[i*100:(i+1)*100,1], color=colors[i], label=f"GS = {guidance_scales[i]}")
    ax.set_title("CLIP Image Embedding PCA")
    # ax.set_title("CLIP Image Embedding TSNE")
    ax.legend()
    # fig.savefig("output/05/05_gs_clip_image_embedding.png")
    plt.show()


    # fig = plt.figure(figsize=(15, 5))
    # for i, item in enumerate(data):
    #     ax = fig.add_subplot(2, 3, i+1)
    #     with io.BytesIO() as buf:
    #         buf.write(item["pic"])
    #         buf.seek(0)
    #         with Image.open(buf) as img:
    #             ax.imshow(img)
    #             ax.set_title(f"guidance_scale={item['guidance_scale']}")
    #             ax.set_axis_off()
    # fig.savefig("output/05/04_1_guidance_image.png")
    # plt.show()

if __name__ == "__main__":
    visualize()
    # with stub.run():
    #     result = main.call()
    #     print(result)
    #     with open("output/05/04_2_intermediate.pkl", "wb") as fp:
    #         pickle.dump(result, fp)