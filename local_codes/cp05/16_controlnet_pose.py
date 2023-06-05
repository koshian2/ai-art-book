from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from PIL import Image
from controlnet_aux import OpenposeDetector, MLSDdetector
import matplotlib.pyplot as plt
from settings import MODEL_DIRECTORY

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

def run_pose_depth(is_mlsd_enable, device="cuda"):
    initial_image = load_resize("data/two_women_are_jumping_on_the_bed.jpg")

    conditional_imgs, control_nets = [], []
    funcs = [run_mlsd, run_pose] if is_mlsd_enable else [run_pose]
    for func in funcs:
        cond, cp = func(initial_image)
        controlnet = ControlNetModel.from_pretrained(cp, torch_dtype=torch.float16)
        control_nets.append(controlnet)
        conditional_imgs.append(cond)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        f"{MODEL_DIRECTORY}/merge_Counterfeit-V3.0_orangemix", controlnet=control_nets, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = lambda images, **kwargs: (images, False)
    pipe.enable_vae_tiling()
    pipe.to(device)
    generator = torch.Generator(device).manual_seed(1235)

    prompt = "two Japanese black gal are jumping on the bed, best quality, extremely detailed"
    negative_prompt = "nsfw, longbody, lowres, bad face, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    second_images = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=50, image=conditional_imgs,
                         generator=generator, num_images_per_prompt=4).images

    if is_mlsd_enable:
        titles = ["condition 1", "result 1", "result 2", "condition 2", "result 3", "result 4"]
        images = [conditional_imgs[0], second_images[0], second_images[1], conditional_imgs[1], second_images[2], second_images[3]]
    else:
        titles = ["original", "result 1", "result 2", "condition", "result 3", "result 4"]
        images = [initial_image, second_images[0], second_images[1], conditional_imgs[0], second_images[2], second_images[3]]
    fig = plt.figure(figsize=(18, 7))
    for i, (im, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(2, 3, i+1)
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(title)
    plt.show()

def main():
    print("-- Disable MLSD --")
    run_pose_depth(False)
    print("-- Enable MLSD --")
    run_pose_depth(True)

if __name__ == "__main__":
    main()