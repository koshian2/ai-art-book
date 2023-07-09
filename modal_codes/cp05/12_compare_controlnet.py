import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import MLSDdetector, OpenposeDetector
import torch
import numpy as np
import matplotlib.pyplot as plt
import modal
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

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


def run_controlnet(original_img, condition_img, controlnet_names, device="cuda"):
    results = [original_img, condition_img]
    prompt = "best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"    

    for pipe_name in ["runwayml/stable-diffusion-v1-5", "NoCrypt/SomethingV2_2"]:
        for controlnet_name in controlnet_names:
            controlnet = ControlNetModel.from_pretrained(
                controlnet_name, torch_dtype=torch.float16, cache_dir=CACHE_DIR
            )
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                pipe_name, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16, cache_dir=CACHE_DIR
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.to(device)
            generator = torch.Generator(device).manual_seed(1234)
            image = pipe(prompt, negative_prompt=negative_prompt, 
                         num_inference_steps=30, generator=generator, image=condition_img).images[0]
            results.append(image)
    return results

def run_canny():
    original_img = load_resize("data/control_condition_01.jpg")
    image = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    image = cv2.Canny(image, 100, 200)
    image = np.stack([image, image, image], axis=-1)
    conditional_img = Image.fromarray(image)
    
    result = run_controlnet(original_img, conditional_img, 
                            ["lllyasviel/sd-controlnet-canny",
                             "lllyasviel/control_v11p_sd15_canny"])
    return result

def run_mlsd():
    mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet', cache_dir=CACHE_DIR)
    original_img = load_resize("data/control_condition_02.jpg")
    conditioning_img = mlsd(original_img)

    result = run_controlnet(original_img, conditioning_img,
                            ["lllyasviel/sd-controlnet-mlsd",
                             "lllyasviel/control_v11p_sd15_mlsd"])
    return result

def run_pose():
    pose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet', cache_dir=CACHE_DIR)
    original_img = load_resize("data/control_condition_03.jpg")
    conditioning_img = pose(original_img)

    result = run_controlnet(original_img, conditioning_img,
                            ["lllyasviel/sd-controlnet-openpose",
                             "lllyasviel/control_v11p_sd15_openpose"])
    return result

@stub.function(
    image=modal.Image.debian_slim().apt_install("libopencv-dev").pip_install(
        "torch", "diffusers[torch]", "transformers", "matplotlib", "controlnet_aux", "opencv-python"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    timeout=600,
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def main():
    fig = plt.figure(figsize=(25, 12))
    for i, mode in enumerate(["Canny", "MLSD", "Pose"]):
        if mode == "Canny":
            result = run_canny()
        elif mode == "MLSD":
            result = run_mlsd()
        elif mode == "Pose":
            result = run_pose()
        titles = ["Orig", mode, "SD1.5-CN1.0", "SD1.5-CN1.1", "ST2V2-CN1.0", "ST2V2-CN1.1"]
        for j in range(len(titles)):
            ax = fig.add_subplot(3, 6, i*6+j+1)
            ax.imshow(result[j])
            ax.axis("off")
            ax.set_title(titles[j])
    fig.savefig(f"{CACHE_DIR}/output/12_compare_controlnet.png")

if __name__ == "__main__":
    with stub.run():
        main.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/12_compare_controlnet.png output', shell=True)
