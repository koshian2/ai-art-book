from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch
import matplotlib.pyplot as plt

def run_local_model(width=512, height=768):
    device = "cuda:1"    
    pipe = StableDiffusionPipeline.from_pretrained(
        "H:/diffusion_models/diffusers/Counterfeit-V3.0", torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = lambda images, **kwargs: (images, False)    
    pipe.to(device)
    prompt = "1girl, look at viewer, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    generator = torch.Generator(device).manual_seed(1234)
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, width=512, height=512, 
                 generator=generator, num_inference_steps=30).images[0]
    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    run_local_model()