import torch
from diffusers import UniPCMultistepScheduler, StableDiffusionImg2ImgPipeline
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw
import modal
import subprocess

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
def i2i_visual_prompt(width=960, height=512):
    # Create Visual Prompt
    with Image.new("RGB", (width, height)) as canvas:
        draw = ImageDraw.Draw(canvas)
        offset = 50
        unit_w = (width-offset*2)//3
        unit_h = height-offset*2
        draw.rectangle([offset, offset, offset+unit_w, offset+unit_h], fill=(255, 69, 0))
        draw.rectangle([offset+unit_w, offset, offset+unit_w*2, offset+unit_h], fill=(0, 191, 255))
        draw.rectangle([offset+unit_w*2, offset, offset+unit_w*3, offset+unit_h], fill=(255, 215, 0))
        canvas = np.array(canvas).astype(np.float32)
    # Add noise
    np.random.seed(1234)
    noise = np.random.randn(height//2, width//2, canvas.shape[2])
    noise = noise.repeat(2, axis=0).repeat(2, axis=1)
    canvas += noise * 20 # 枠の部分にも弱くノイズをかける
    canvas[offset:-offset, offset:-offset, :] += noise[offset:-offset, offset:-offset, :] * 80 # メインの部分は強くかける
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(canvas)

    # Image2Image
    device = "cuda"    
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_tiling()
    pipe.to(device)

    prompt = "3girls, little magical girls, looking at viewer, upper body, masterpiece, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, " \
                    "fewer digits, cropped, worst quality, low quality"

    generator = torch.Generator().manual_seed(1234)
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, generator=generator,
                 num_inference_steps=50, output_type="pil", image=canvas, strength=1).images[0]
    image.save(f"{CACHE_DIR}/output/15_i2i_visual_prompt.png")

if __name__ == "__main__":
    with stub.run():
        i2i_visual_prompt.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/15_i2i_visual_prompt.png output', shell=True)
