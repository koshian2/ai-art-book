from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch
import modal
import io
import pickle

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

@stub.function(
    image=modal.Image.debian_slim().pip_install("torch", "diffusers[torch]", "transformers"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
)
def run_stable_diffusion_modal(prompt,
                               model_name,
                               seed):
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16,
        cache_dir=CACHE_DIR)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_tiling()
    pipe.to(device)

    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    generator = torch.Generator(device).manual_seed(seed)
    image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=30, generator=generator, num_images_per_prompt=4).images

    with io.BytesIO() as buf:
        pickle.dump(image, buf)
        return buf.getvalue()

def run_stable_diffusion(prompt,
                         model_name,
                         seed):
    with stub.run():
        img_bytes = run_stable_diffusion_modal.call(prompt, model_name, seed)
        with io.BytesIO() as buf:
            buf.write(img_bytes)
            buf.seek(0)
            data = pickle.load(buf)
            return data
            
def make_ui():
    import gradio as gr

    prompt = gr.Textbox()
    model_name = gr.Radio(choices=["stabilityai/stable-diffusion-2-1-base", "NoCrypt/SomethingV2_2"], value="stabilityai/stable-diffusion-2-1-base")
    seed = gr.Slider(minimum=1, maximum=256*256, randomize=True, step=1)
    gallery = gr.Gallery().style(grid=[2], height="auto", container=True)
    demo = gr.Interface(run_stable_diffusion, [prompt, model_name, seed], gallery)
    return demo
        
if __name__ == "__main__":
    demo = make_ui()
    demo.launch()