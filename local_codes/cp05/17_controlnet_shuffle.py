from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from PIL import Image
from controlnet_aux import ContentShuffleDetector
import matplotlib.pyplot as plt

def load_resize(path):
    # 省略

def run_shuffle(device="cuda:1"):
    initial_image = load_resize("data/shibuya_crossing.jpg")
    processor = ContentShuffleDetector()
    control_image = processor(initial_image)

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11e_sd15_shuffle",
                                                  torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "H:/diffusion_models/diffusers/merge_Counterfeit-V3.0_orangemix", 
        controlnet=controlnet, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = lambda images, **kwargs: (images, False)
    pipe.enable_vae_tiling()
    pipe.to(device)
    generator = torch.Generator(device).manual_seed(1234)

    prompt = "city, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad face, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    ## 以下略
    
if __name__ == "__main__":
    run_shuffle()
