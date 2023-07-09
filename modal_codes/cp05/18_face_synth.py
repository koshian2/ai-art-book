from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from PIL import Image
import matplotlib.pyplot as plt
import glob
from safetensors.torch import load_file
from laion_face_common import generate_annotation
import matplotlib.pyplot as plt
from utils import load_safetensors_lora
from settings import LORA_DIRECTORY
import shutil
import modal
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

# Kaggleからダウンロードしたarchive.zipをdataフォルダに入れてください

def extract_dataset():
    shutil.unpack_archive("data/archive.zip", "data")
    shutil.move("data/test", "data/fer2013")

def load_fer2013():
    extract_dataset()
    dataset = {}
    for dir in sorted(glob.glob("data/fer2013/*")):
        files = sorted([x.replace("\\", "/") for x in glob.glob(dir+"/*")])
        emotion = dir.replace("\\", "/").split("/")[-1]
        annotations = []
        for f in files:
            try:
                anno = generate_annotation(Image.open(f).resize((512, 512), Image.Resampling.BICUBIC).convert("RGB"), 1)
            except:
                print("skip", f)
                continue
            else:
                print("success", f)
                annotations.append(anno)
            if len(annotations) == 6:
                break
        dataset[emotion] = annotations
    return dataset

def show_dataset(dataset):
    dataset = load_fer2013()
    fig = plt.figure(figsize=(15, 15))
    for i, (emotion, imgs) in enumerate(dataset.items()):
        for j, img in enumerate(imgs):
            ax = fig.add_subplot(7, 6, i*6+j+1)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{emotion} {j+1}")
    fig.savefig(f"{CACHE_DIR}/output/18_face_synth_dataset.png")

@stub.function(
    image=modal.Image.debian_slim().apt_install("libopencv-dev").pip_install(
        "torch", "diffusers[torch]", "transformers", "matplotlib", "controlnet_aux", "opencv-python", "mediapipe"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    timeout=600,
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def run_pose_mediapipe_face(device="cuda"):
    controlnet = ControlNetModel.from_pretrained("CrucibleAI/ControlNetMediaPipeFace", 
                                                 subfolder="diffusion_sd15", torch_dtype=torch.float16, cahce_dir=CACHE_DIR)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", controlnet=controlnet, torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe = load_safetensors_lora(pipe, f"{LORA_DIRECTORY}/lumine1-000008.safetensors", alpha=0.6)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    negative_prompt = "glasses, longbody, lowres, bad face, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    dataset = load_fer2013()
    show_dataset(dataset)
    fig = plt.figure(figsize=(15, 15))

    for i, (emotion, anno_imgs) in enumerate(dataset.items()):
        for j, anno in enumerate(anno_imgs):
            generator = torch.Generator(device).manual_seed(1234)
            prompt = f"luminedef, luminernd, 1girl, face, {emotion}, best quality, extremely detailed"
            images = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=30, image=Image.fromarray(anno),
                          generator=generator).images[0]

            ax = fig.add_subplot(7, 6, i*6+j+1)
            ax.imshow(images)
            ax.axis("off")
            ax.set_title(f"{emotion} {j+1}")
    fig.savefig(f"{CACHE_DIR}/output/18_face_synth_result.png")
    
if __name__ == "__main__":
    with stub.run():
        run_pose_mediapipe_face.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/18_face_synth* .', shell=True)
