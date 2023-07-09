import torch
import imageio
from diffusers import UniPCMultistepScheduler, ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor
from PIL import Image
import numpy as np
import modal
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

def run_text2video_controlnet(model_id, prompt, 
                              conditional_video_path, 
                              output_video_path):
    device = "cuda"
    # Read conditional video
    reader = imageio.get_reader(conditional_video_path, "ffmpeg")
    pose_images = [Image.fromarray(img) for img in reader.iter_data()]

    # Control net
    controlnet_id = "lllyasviel/control_v11p_sd15_openpose"
    controlnet = ControlNetModel.from_pretrained(
        controlnet_id, torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=torch.float16, cache_dir=CACHE_DIR
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # Set attantion processor
    pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
    pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to(device)

    # Copy latents accross the frames
    generator = torch.Generator().manual_seed(1234)
    latents = torch.randn((1, 4, 64, 64), generator=generator).repeat(len(pose_images), 1, 1, 1)
    latents = latents.to(device, torch.float16)

    # Generate video
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    result = pipe(prompt=[prompt]*len(pose_images), 
                  negative_prompt=[negative_prompt]*len(pose_images),
                  image=pose_images, latents=latents, output_type="latent").images
    # Decode frame by frame to avoid CUDA OOM
    frames = []
    with torch.no_grad():
        for i in range(len(result)):
            img = pipe.decode_latents(result[i:i+1])
            frames.append((img[0] * 255.0).astype(np.uint8))
    imageio.mimsave(output_video_path, frames, fps=4)

    # save frames
    n = max(len(frames) // 4, 1)
    joint_images = np.stack(frames[::n][:4], axis=1)
    joint_images = joint_images.reshape(joint_images.shape[0], -1, joint_images.shape[3])
    with Image.fromarray(joint_images) as img:
        img.save(output_video_path.replace(".mp4", "_frames.jpg"), quality=92)

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "transformers", "imageio[ffmpeg, pyav]", "xformers"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="a10g",
    timeout=600,
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def main():
    run_text2video_controlnet("runwayml/stable-diffusion-v1-5",
                              "a stormtrooper dancing on the beach, best quality, extremely detailed",
                              "data/dance1_corr.mp4", 
                              f"{CACHE_DIR}/output/10_text2video_controlnet_1.mp4")
    run_text2video_controlnet("NoCrypt/SomethingV2_2",
                              "Hatsune Miku is dancing on a snowfield, best quality, extremely detailed",
                              "data/dance2_corr.mp4", 
                              f"{CACHE_DIR}/output/10_text2video_controlnet_2.mp4")
    run_text2video_controlnet("NoCrypt/SomethingV2_2",
                              "a girl is dancing on a classroom, 1girl, purple hair, best quality, extremely detailed",
                              "data/dance3_corr.mp4", 
                              f"{CACHE_DIR}/output/10_text2video_controlnet_3.mp4")
    
if __name__ == "__main__":
    with stub.run():
        main.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/10_text2video_controlnet* .', shell=True)
    
