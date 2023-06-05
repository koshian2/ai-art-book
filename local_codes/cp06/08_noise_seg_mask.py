import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def noise_seg_mask(width=960, height=512):
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    prompts = "masterpiece, best quality, 1girl, brown-haired girl, autumn leaves, standing, beauty, harmony, nature, breathtaking, foliage, dancing, picturesque, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    generator = torch.Generator().manual_seed(1234)
    # latent value
    randn = torch.randn((1, 4, height//8, width//8), generator=generator).to(device, torch.float16)
    # initial run
    initial_image = pipe(prompt=prompts, negative_prompt=negative_prompt, 
                         num_inference_steps=50, output_type="pil", latents=randn).images[0]
    
    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
    pixel_values = image_processor(initial_image, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = image_segmentor(pixel_values)
    seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[initial_image.size[::-1]])[0]

    seg_array = np.array(seg)
    # ID:12 = person
    seg_mask = np.zeros_like(seg_array, dtype=np.uint8)
    seg_mask[seg_array==12] = 255 # person mask
    with Image.fromarray(seg_mask) as mask:
        seg_mask = mask.resize((width//8, height//8), Image.Resampling.BOX)
        seg_mask = np.stack([np.array(seg_mask) for i in range(4)], axis=0)[None]
        seg_mask = torch.from_numpy(seg_mask / 255.0).to(device, torch.float16)
    randn_segmask = randn + randn * seg_mask * 0.3
    second_image = pipe(prompt=prompts, negative_prompt=negative_prompt, 
                         num_inference_steps=50, output_type="pil", latents=randn_segmask).images[0]
    
    fig = plt.figure(figsize=(18, 7))
    for i, img in enumerate([initial_image, second_image]):
        ax = fig.add_subplot(1, 2, i+1)
        ax.imshow(img)
        ax.set_title(("w/o" if i == 0 else "with") + " segmentation mask")
        ax.axis("off")
    plt.show()

if __name__ == "__main__":
    noise_seg_mask()
