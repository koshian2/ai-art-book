from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

ada_palette = np.asarray(...) # 長いので省略

def load_resize(path):
    img = Image.open(path)
    if img.width > img.height:
        height = 512
        width = int((img.width / img.height * 512) // 8 * 8)
    else:
        width = 512
        height = int((img.height / img.width * 512) // 8 * 8)
    img = img.resize((width, height), Image.Resampling.BICUBIC)
    return img

def run_depth(device="cuda:1"):
    # Load image
    intial_image = load_resize("data/shibuya_crossing.jpg")

    # segmentation
    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-swin-large")
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-swin-large")

    pixel_values = image_processor(intial_image, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = image_segmentor(pixel_values)
    seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[intial_image.size[::-1]])[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label, color in enumerate(ada_palette):
        color_seg[seg == label, :] = color
    color_seg = color_seg.astype(np.uint8)
    control_image = Image.fromarray(color_seg)

    # second run
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)
    generator = torch.Generator(device).manual_seed(1234)
    prompt = "Otakus in the crowd running at comiket in the Roman Empire era, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"    

    second_images = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=30, image=control_image,
                         generator=generator, width=960, height=512, num_images_per_prompt=4).images
    
    titles = ["original", "result 1", "result 2", "condition", "result 3", "result 4"]
    images = [intial_image, second_images[0], second_images[1], control_image, second_images[2], second_images[3]]
    fig = plt.figure(figsize=(20, 7))
    for i, (im, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(2, 3, i+1)
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(title)
    plt.show()

if __name__ == "__main__":
    run_depth()
