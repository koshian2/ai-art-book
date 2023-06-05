from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionPipeline
from controlnet_aux import MLSDdetector, LineartDetector, LineartAnimeDetector, HEDdetector, PidiNetDetector
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def run_canny(input_image, low_threshold=100, high_threshold=200):
    checkpoint = "lllyasviel/control_v11p_sd15_canny"

    image = np.array(input_image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = np.stack([image, image, image], axis=-1)
    control_image = Image.fromarray(image)

    return control_image, checkpoint

def run_mlsd(input_image):
    checkpoint = "lllyasviel/control_v11p_sd15_mlsd"

    processor = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
    control_image = processor(input_image)
    return control_image, checkpoint

def run_lineart(input_image):
    checkpoint = "lllyasviel/control_v11p_sd15_lineart"

    processor = LineartDetector.from_pretrained("lllyasviel/Annotators")    
    control_image = processor(input_image)
    return control_image, checkpoint

def run_lineart_anime(input_image):
    checkpoint = "lllyasviel/control_v11p_sd15s2_lineart_anime"

    processor = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
    control_image = processor(input_image)
    return control_image, checkpoint

def run_scribble(input_image):
    checkpoint = "lllyasviel/control_v11p_sd15_scribble"

    processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
    control_image = processor(input_image)
    return control_image, checkpoint

def run_softedge(input_image):
    checkpoint = "lllyasviel/control_v11p_sd15_softedge"

    processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
    control_image = processor(input_image)
    return control_image, checkpoint

def run_all(prompt, source_img_path, device="cuda"):
    initial_image = Image.open(source_img_path)

    modes = ["canny", "mlsd", "lineart", "lineart_anime", "scribble", "softedge"]
    functions = [run_canny, run_mlsd, run_lineart, run_lineart_anime, run_scribble, run_softedge]
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"    

    conditions, generated = [], []
    for i, mode in enumerate(modes):
        cond_img, checkpoint = functions[i](initial_image)
        conditions.append(cond_img)

        controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "NoCrypt/SomethingV2_2", controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_vae_tiling()
        pipe.to(device)

        generator = torch.Generator(device).manual_seed(1234)
        second_images = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=30, image=cond_img,
                            generator=generator, width=cond_img.width, height=cond_img.height).images[0]
        generated.append(second_images)

    generated = conditions + generated
    fig = plt.figure(figsize=(18, 10))
    n = len(modes)
    for i in range(len(generated)):
        ax = fig.add_subplot(2, n, i+1)
        ax.imshow(generated[i])
        ax.axis("off")
        ax.set_title(modes[i % n] + (" cond" if i // n == 0 else " generated"))
    plt.show()

def main():
    prompt = "1girl, look at viewer, best quality, extremely detailed"
    run_all(prompt, "data/black_hair_girl.png")
    prompt = "the building where the god who must be defeated as the last boss lives, best quality, extremely detailed"
    run_all(prompt, "data/building.jpg")

if __name__ == "__main__":
    main()