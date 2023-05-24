from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch
import gradio as gr

def run_stable_diffusion(prompt,
                         model_name,
                         seed, state):
    device = "cuda"
    if state["model_name"] != model_name or state["pipe"] is None:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name, torch_dtype=torch.float16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.vae.enable_tiling()
        pipe.to(device)
        state["pipe"] = pipe
        state["model_name"] = model_name
    else:
        pipe = state["pipe"]

    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    generator = torch.Generator(device).manual_seed(seed)
    image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=30, generator=generator, num_images_per_prompt=4).images
    return image, state

def make_ui():
    prompt = gr.Textbox()
    model_name = gr.Radio(choices=["stabilityai/stable-diffusion-2-1-base", "NoCrypt/SomethingV2_2"], value="stabilityai/stable-diffusion-2-1-base")
    seed = gr.Slider(minimum=1, maximum=256*256, randomize=True, step=1)
    state = gr.State({
        "pipe": None,
        "model_name": ""
    })
    gallery = gr.Gallery().style(grid=[2], height="auto", container=True)
    demo = gr.Interface(run_stable_diffusion, [prompt, model_name, seed, state], [gallery, state])
    return demo
        
if __name__ == "__main__":
    demo = make_ui()
    demo.launch()