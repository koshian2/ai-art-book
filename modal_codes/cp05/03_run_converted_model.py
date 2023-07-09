from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch
import matplotlib.pyplot as plt
import modal
import subprocess
import os
from huggingface_hub import hf_hub_download

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

def model_convert():
    hf_hub_download(repo_id="gsdf/Counterfeit-V3.0", filename="Counterfeit-V3.0_fp16.safetensors",
                    local_dir=f"{CACHE_DIR}/safetensors")
    subprocess.run(
        "python convert/convert_original_stable_diffusion_to_diffusers.py "\
             "--original_config_file convert/v1-inference.yaml "\
            f"--checkpoint_path {CACHE_DIR}/safetensors/Counterfeit-V3.0_fp16.safetensors "\
             "--image_size 512 "\
             "--prediction_type epsilon "\
             "--extract_ema "\
             "--upcast_attention "\
            f"--dump_path {CACHE_DIR}/Counterfeit-V3.0 "\
             "--from_safetensors "\
             "--device cpu", shell=True)

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "transformers", "matplotlib", "omegaconf"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def run_local_model(width=512, height=768):
    # download and convert
    model_convert()
    # Loading
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        f"{CACHE_DIR}/Counterfeit-V3.0", torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    prompt = "1girl, look at viewer, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    generator = torch.Generator(device).manual_seed(1234)
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, width=width, height=height, 
                 generator=generator, num_inference_steps=30).images[0]
    plt.imshow(image)

    output_filename = f"{CACHE_DIR}/output/03_run_converted_model.png"
    os.makedirs(os.path.basename(output_filename), exist_ok=True)
    plt.savefig(output_filename)

if __name__ == "__main__":
    with stub.run():
        run_local_model.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/03_run_converted_model* .', shell=True)
    