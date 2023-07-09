from diffusers import DiffusionPipeline, UniPCMultistepScheduler
import torch
import matplotlib.pyplot as plt
from utils import load_safetensors_lora
from settings import MODEL_DIRECTORY, LORA_DIRECTORY
import modal
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "transformers", "matplotlib"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def change_cloth(width=512, height=960):
    device = "cuda"
    models = [
        "NoCrypt/SomethingV2_2",
        f"{MODEL_DIRECTORY}/merge_Counterfeit-V3.0_orangemix"
    ]
    model_names = ["SomethingV2_2", "merge", "matplotlib"]
    
    fig = plt.figure(figsize=(18, 14))
    cloths = ["", "yukata", "basketball uniform", "track and field uniform", "bikini"]
    weights = [1.0, 1.1, 1.8, 1.55, 1.95]
    for i, model in enumerate(models):
        for j, (cloth, weight) in enumerate(zip(cloths, weights)):
            pipe = DiffusionPipeline.from_pretrained(
                model, torch_dtype=torch.float16,
                custom_pipeline="lpw_stable_diffusion", safety_checker=None, cache_dir=CACHE_DIR)
            pipe = load_safetensors_lora(pipe, f"{LORA_DIRECTORY}/lumine1-000008.safetensors", alpha=0.3)
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_vae_tiling()
            pipe.to(device)

            prompt = "luminedef, luminernd, "
            prompt += f"({cloth}:{weight}), " if cloth != "" else ""
            prompt += "1girl, look at viewer, best quality, extremely detailed"
            negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
            generator = torch.Generator(device).manual_seed(1234)
            image = pipe(prompt=prompt, negative_prompt=negative_prompt, max_embeddings_multiples=3,
                        generator=generator, width=width, height=height, num_inference_steps=30).images[0]
            
            ax = fig.add_subplot(2, 5, 5*i+j+1)
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(f"{model_names[i]} {cloth if cloth != '' else 'None'}")

    fig.savefig(f"{CACHE_DIR}/output/09_lora_change_prompt.png")

if __name__ == "__main__":
    with stub.run():
        change_cloth.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/09_lora_change_prompt.png output', shell=True)
