import copy
import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler, AutoencoderKL
import modal
import subprocess
from huggingface_hub import hf_hub_download

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

def model_convert():
    hf_hub_download(repo_id="WarriorMama777/OrangeMixs", filename="AOM3A1B_orangemixs.safetensors",
                    subfolder="Models/AbyssOrangeMix3",
                    local_dir=f"{CACHE_DIR}/safetensors")
    subprocess.run(
        "python convert/convert_original_stable_diffusion_to_diffusers.py "\
             "--original_config_file convert/v1-inference.yaml "\
            f"--checkpoint_path {CACHE_DIR}/safetensors/Models/AbyssOrangeMix3/AOM3A1B_orangemixs.safetensors "\
             "--image_size 512 "\
             "--prediction_type epsilon "\
             "--extract_ema "\
             "--upcast_attention "\
            f"--dump_path {CACHE_DIR}/AOM3A1B_orangemixs "\
             "--from_safetensors "\
             "--device cpu", shell=True)
    hf_hub_download(repo_id="hakurei/waifu-diffusion-v1-4", filename="kl-f8-anime2.ckpt",
                    subfolder="vae",
                    local_dir=f"{CACHE_DIR}/safetensors")
    subprocess.run(
        "python convert/convert_vae_pt_to_diffusers.py "\
            f"--vae_pt_path {CACHE_DIR}/safetensors/vae/kl-f8-anime2.ckpt "\
            f"--dump_path {CACHE_DIR}/vae_kl-f8-anime2", shell=True)

def merge_network(pipe_source, pipe_merge, attr, ratio):
    merge_net = copy.deepcopy(getattr(pipe_source, attr))
    pipe_source_params = dict(getattr(pipe_source, attr).named_parameters())
    pipe_merge_params = dict(getattr(pipe_merge, attr).named_parameters())
    for key, param in merge_net.named_parameters():
        x = pipe_source_params[key] * (1-ratio) + pipe_merge_params[key] * ratio
        param.data = x
    return merge_net

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "transformers", "matplotlib", "omegaconf", "pytorch_lightning"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def merge_and_save():
    model_convert()
    pipe_orange = StableDiffusionPipeline.from_pretrained(
        f"{CACHE_DIR}/AOM3A1B_orangemixs", torch_dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        f"{CACHE_DIR}/Counterfeit-V3.0", torch_dtype=torch.float16)
    pipe.text_encoder = merge_network(pipe, pipe_orange, "text_encoder", 0.25)
    pipe.unet = merge_network(pipe, pipe_orange, "unet", 0.25)
    pipe.vae =  AutoencoderKL.from_pretrained(
        f"{CACHE_DIR}/vae_kl-f8-anime2", torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.save_pretrained(f"{CACHE_DIR}/merge_Counterfeit-V3.0_orangemix")

if __name__ == "__main__":
    with stub.run():
        merge_and_save.call()