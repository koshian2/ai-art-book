from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import load_resize, ade_palette
import modal
import subprocess
import os

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "transformers", "matplotlib"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def run_seg(device="cuda"):
    # Load image
    intial_image = load_resize("data/shibuya_crossing.jpg")

    # segmentation
    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-swin-large", cache_dir=CACHE_DIR)
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-swin-large", cache_dir=CACHE_DIR)

    pixel_values = image_processor(intial_image, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = image_segmentor(pixel_values)
    seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[intial_image.size[::-1]])[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
    for label, color in enumerate(ade_palette):
        color_seg[seg == label, :] = color
    color_seg = color_seg.astype(np.uint8)
    control_image = Image.fromarray(color_seg)

    # second run
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", controlnet=controlnet, torch_dtype=torch.float16, cache_dir=CACHE_DIR
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
    fig.savefig(f"{CACHE_DIR}/output/15_controlnet_seg.png")

if __name__ == "__main__":
    with stub.run():
        run_seg.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/15_controlnet_seg.png output', shell=True)
