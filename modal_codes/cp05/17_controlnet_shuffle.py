from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import ContentShuffleDetector
import matplotlib.pyplot as plt
from utils import load_resize
from settings import MODEL_DIRECTORY
import modal
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

@stub.function(
    image=modal.Image.debian_slim().apt_install("libopencv-dev").pip_install(
        "torch", "diffusers[torch]", "transformers", "matplotlib", "controlnet_aux", "opencv-python"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    timeout=600,
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def run_shuffle(device="cuda"):
    initial_image = load_resize("data/shibuya_crossing.jpg")
    processor = ContentShuffleDetector()
    control_image = processor(initial_image)

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11e_sd15_shuffle",
                                                  torch_dtype=torch.float16, cache_dir=CACHE_DIR)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        f"{MODEL_DIRECTORY}/merge_Counterfeit-V3.0_orangemix", safety_checker=None, 
        controlnet=controlnet, torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)
    generator = torch.Generator(device).manual_seed(1234)

    prompt = "city, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad face, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    second_images = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=50, image=control_image,
                         generator=generator, num_images_per_prompt=4).images

    titles = ["original", "result 1", "result 2", "condition", "result 3", "result 4"]
    images = [initial_image, second_images[0], second_images[1], control_image, second_images[2], second_images[3]]
    fig = plt.figure(figsize=(18, 7))
    for i, (im, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(2, 3, i+1)
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(title)
    fig.savefig(f"{CACHE_DIR}/output/17_controlnet_shuffle.png")
    
if __name__ == "__main__":
    with stub.run():
        run_shuffle.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/17_controlnet_shuffle.png output', shell=True)
    