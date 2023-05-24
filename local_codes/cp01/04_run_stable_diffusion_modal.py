import io
import modal

stub = modal.Stub()

@stub.function(
    image=modal.Image.debian_slim().pip_install("torch", "diffusers[torch]", "transformers", "accelerate"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
)
async def run_stable_diffusion(prompt: str):
    import torch
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=torch.float16
    )
    pipe.to("cuda")

    generator = torch.Generator("cuda").manual_seed(1234)
    image = pipe(prompt, num_inference_steps=30, generator=generator).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    return img_bytes

if __name__ == "__main__":
    with stub.run():
        img_bytes = run_stable_diffusion.call("Corgi riding a bike in Times Square")
        with open("output.png", "wb") as f:
            f.write(img_bytes)