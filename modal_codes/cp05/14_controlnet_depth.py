from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionPipeline
from controlnet_aux import MidasDetector
import torch
import matplotlib.pyplot as plt
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
def run_depth(device="cuda"):
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    prompt = "panorama of floating continent, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    generator = torch.Generator(device).manual_seed(1234)
    intial_image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=30,
                        generator=generator, width=960, height=512).images[0]

    # condition image
    midas = MidasDetector.from_pretrained("lllyasviel/Annotators", cache_dir=CACHE_DIR)
    depth_map = midas(intial_image)

    # second run
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", controlnet=controlnet, torch_dtype=torch.float16,
        cache_dir=CACHE_DIR        
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    generator = torch.Generator(device).manual_seed(1234)
    second_images = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=30, image=depth_map,
                         generator=generator, width=960, height=512, num_images_per_prompt=4).images
    
    titles = ["original", "result 1", "result 2", "condition", "result 3", "result 4"]
    images = [intial_image, second_images[0], second_images[1], depth_map, second_images[2], second_images[3]]
    fig = plt.figure(figsize=(20, 7))
    for i, (im, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(2, 3, i+1)
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(title)
    fig.savefig(f"{CACHE_DIR}/output/14_controlnet_depth.png")

if __name__ == "__main__":
    with stub.run():
        run_depth.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/14_controlnet_depth.png output', shell=True)
