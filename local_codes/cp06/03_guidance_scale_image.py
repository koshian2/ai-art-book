import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import matplotlib.pyplot as plt

def guidance_scale_compare(width=960, height=512):
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    result = []
    for gs in [1.5, 3, 7, 10, 20, 50]:
        generator = torch.Generator().manual_seed(1234)
        prompts = "a girl standing on the beach, smiling and having fun, blue haired, 1girl, border skirt, denim jacket, best quality, extremely detailed"
        negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
        image = pipe(prompt=prompts, negative_prompt=negative_prompt, generator=generator,
                    num_inference_steps=50, output_type="pil", width=width, height=height,
                    guidance_scale=gs).images[0]
        item = {"guidance_scale": gs, "pic": image}
        result.append(item)

    fig = plt.figure(figsize=(15, 5))
    for i, item in enumerate(result):
        ax = fig.add_subplot(2, 3, i+1)
        ax.imshow(item["pic"])
        ax.set_title(f"guidance_scale={item['guidance_scale']}")
        ax.set_axis_off()
    plt.show()

if __name__ == "__main__":
    guidance_scale_compare()