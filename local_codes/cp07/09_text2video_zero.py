import torch
import imageio
from diffusers import TextToVideoZeroPipeline, UniPCMultistepScheduler
from PIL import Image

def run_text2video(model_id, prompt, output_video_path, seed=1234):
    device = "cuda"
    pipe = TextToVideoZeroPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to(device)

    generator = torch.Generator(device).manual_seed(1234)
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    result = pipe(prompt=prompt, negative_prompt=negative_prompt, generator=generator).images
    result = [(r * 255).astype("uint8") for r in result]

    # save video
    imageio.mimsave(output_video_path, result, fps=4)
    # save first, last frame
    for i in [0, -1]:
        with Image.fromarray(result[i]) as img:
            if i == 0:
                out_img_path = output_video_path.replace(".mp4", "_first.jpg")
            else:
                out_img_path = output_video_path.replace(".mp4", "_last.jpg")
            img.save(out_img_path, quality=92)

def main():
    run_text2video("runwayml/stable-diffusion-v1-5",
                   "a panda playing a guitar in times square, best quality, extremely detailed",
                   "output/09_text2video_1.mp4")
    run_text2video("NoCrypt/SomethingV2_2",
                   "a girl is invoking flame magic from her wand, 1girl, purple hair, best quality, extremely detailed",
                   "output/09_text2video_2.mp4")

if __name__ == "__main__":
    main()