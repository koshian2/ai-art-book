import torch
from diffusers import UniPCMultistepScheduler, StableDiffusionPipeline
from utils import load_safetensors_lora, LORA_PATH
from omnidirectional_viewer import omniview

def main():
    device = "cuda"
    model_id = "NoCrypt/SomethingV2_2"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16)
    pipe.enable_vae_tiling()
    pipe = load_safetensors_lora(
        pipe, f"{LORA_PATH}/LatentLabs360.safetensors", alpha=1.0
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device, torch.float16)

    generator = torch.Generator().manual_seed(1234)
    prompt = "a 360 equirectangular panorama, modelshoot style, bedroom in a rich mansion, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, 
                 generator=generator, width=1024, height=512,
                 num_inference_steps=30).images[0]
    image.save("output/11_360dregree.png")
    # ESCボタンで終了
    omniview("output/11_360dregree.png", width=640, height=360)

if __name__ == "__main__":
    main()
