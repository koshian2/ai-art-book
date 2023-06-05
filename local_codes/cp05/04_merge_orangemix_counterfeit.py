import copy
import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler, AutoencoderKL
from settings import MODEL_DIRECTORY

## settings.pyの MODEL_DIRECTORY にDiffusersモデルを保存したディレクトリを記述

def merge_network(pipe_source, pipe_merge, attr, ratio):
    merge_net = copy.deepcopy(getattr(pipe_source, attr))
    pipe_source_params = dict(getattr(pipe_source, attr).named_parameters())
    pipe_merge_params = dict(getattr(pipe_merge, attr).named_parameters())
    for key, param in merge_net.named_parameters():
        x = pipe_source_params[key] * (1-ratio) + pipe_merge_params[key] * ratio
        param.data = x
    return merge_net

def merge_and_save():
    pipe_orange = StableDiffusionPipeline.from_pretrained(
        f"{MODEL_DIRECTORY}/AOM3A1B_orangemixs", torch_dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        f"{MODEL_DIRECTORY}/Counterfeit-V3.0", torch_dtype=torch.float16)
    pipe.text_encoder = merge_network(pipe, pipe_orange, "text_encoder", 0.25)
    pipe.unet = merge_network(pipe, pipe_orange, "unet", 0.25)
    pipe.vae =  AutoencoderKL.from_pretrained(
        f"{MODEL_DIRECTORY}/vae_kl-f8-anime2", torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.save_pretrained(f"{MODEL_DIRECTORY}/merge_Counterfeit-V3.0_orangemix")

if __name__ == "__main__":
    merge_and_save()