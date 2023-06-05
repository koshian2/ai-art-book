from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import ContentShuffleDetector
import matplotlib.pyplot as plt
from utils import load_resize
from settings import MODEL_DIRECTORY

def run_shuffle(device="cuda"):
    initial_image = load_resize("data/shibuya_crossing.jpg")
    processor = ContentShuffleDetector()
    control_image = processor(initial_image)

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11e_sd15_shuffle",
                                                  torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        f"{MODEL_DIRECTORY}/merge_Counterfeit-V3.0_orangemix", 
        controlnet=controlnet, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = lambda images, **kwargs: (images, False)
    pipe.enable_vae_tiling()
    pipe.to(device)
    generator = torch.Generator(device).manual_seed(1234)

    prompt = "city, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad face, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    second_images = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=50, image=control_image,
                         generator=generator, num_images_per_prompt=4).images

    titles = ["original", "result 1", "result 2", "condition", "result 3", "result 4"]
    images = [initial_image, second_images[0], second_images[1], control_image, second_images[2], second_images[3]]
    fig = plt.figure(figsize=(18, 7))
    for i, (im, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(2, 3, i+1)
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(title)
    plt.show()
    
if __name__ == "__main__":
    run_shuffle()
