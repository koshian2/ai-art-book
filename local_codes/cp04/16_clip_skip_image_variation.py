from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from transformers import CLIPVisionModelWithProjection, AutoProcessor, CLIPTextModel
import torch
import matplotlib.pyplot as plt
import numpy as np

def clip_skip_image_variation():
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained("JosephusCheung/ACertainModel", torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker=lambda images, **kwargs: (images, False)
    pipe.vae.enable_tiling()
    pipe.to(device)

    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    prompt = "Smiling girl with Christmas tree in background, green hair, knit sweater, night view, best quality"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    result = {}
    for i in range(8):
        image_embeddings = []
        pipe.text_encoder = CLIPTextModel.from_pretrained("JosephusCheung/ACertainModel", 
                                                        subfolder="text_encoder", 
                                                        num_hidden_layers=11-i, torch_dtype=torch.float16).to(device)
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
    plt.bar(result.keys(), result.values())
    plt.ylim((values.min()*0.99, values.max()*1.01))
    plt.show()
    print(result)

if __name__ == "__main__":
    clip_skip_image_variation()
