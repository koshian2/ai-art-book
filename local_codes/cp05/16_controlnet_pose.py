from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from PIL import Image
from controlnet_aux import OpenposeDetector, MLSDdetector
import matplotlib.pyplot as plt

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

def run_pose(initial_img):
    checkpoint = "lllyasviel/control_v11p_sd15_openpose"
    pose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    conditioning_img = pose(initial_img, hand_and_face='Full')
    return conditioning_img, checkpoint

def run_mlsd(initial_img):
    checkpoint = "lllyasviel/control_v11p_sd15_mlsd"
    processor = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    conditioning_img = processor(initial_img)
    return conditioning_img, checkpoint

def run_pose_depth(is_mlsd_enable, device="cuda:1"):
    initial_image = load_resize("data/two_women_are_jumping_on_the_bed.jpg")

    conditional_imgs, control_nets = [], []
    funcs = [run_mlsd, run_pose] if is_mlsd_enable else [run_pose]
    for func in funcs:
        cond, cp = func(initial_image)
        controlnet = ControlNetModel.from_pretrained(cp, torch_dtype=torch.float16)
        control_nets.append(controlnet)
        conditional_imgs.append(cond)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "H:/diffusion_models/diffusers/merge_Counterfeit-V3.0_orangemix", controlnet=control_nets, torch_dtype=torch.float16)
    # 以下省略
