import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from transformers import CLIPTextModel
import matplotlib.pyplot as plt

def get_indices(pipe, prompt: str):
    """Utility function to list the indices of the tokens you wish to alte"""
    ids = pipe.tokenizer(prompt).input_ids
    indices = {i: tok for tok, i in zip(pipe.tokenizer.convert_ids_to_tokens(ids), range(len(ids)))}
    return indices

def main():
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained("JosephusCheung/ACertainModel", torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_tiling()
    pipe.to(device)

    prompt = "Smiling girl with Christmas tree in background, green hair, knit sweater, night view, best quality"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    print(get_indices(pipe, prompt))

    images, norms = [], []
    for i in range(8):
        pipe.text_encoder = CLIPTextModel.from_pretrained("JosephusCheung/ACertainModel", 
                                                          subfolder="text_encoder", 
                                                          num_hidden_layers=11-i, torch_dtype=torch.float16).to(device)
        generator = torch.Generator(device).manual_seed(1234)
        prompt_embed = pipe._encode_prompt(prompt, device, negative_prompt=negative_prompt, num_images_per_prompt=1, do_classifier_free_guidance=True)
        image = pipe(prompt_embeds=prompt_embed[1:2], negative_prompt_embeds=prompt_embed[0:1], generator=generator, num_inference_steps=30,
                     width=960).images[0]
        images.append(image)
        norm = prompt_embed[1].std(dim=-1)
        norms.append(norm)

    fig = plt.figure(figsize=(18, 10))
    for i in range(len(images)):
        ax = fig.add_subplot(2, 4, i+1)
        ax.imshow(images[i])
        ax.set_title(f"Clip skip = {i+1}")
        ax.axis("off")
    plt.show()
    fig = plt.figure(figsize=(18, 8))
    for i in range(len(norms)):
        ax = fig.add_subplot(2, 4, i+1)
        ax.plot(norms[i][:20].detach().cpu().numpy())
        ax.set_title(f"Clip skip = {i+1}")
        ax.set_ylim((1.0, 1.2))
    plt.show()

if __name__ == "__main__":
    main()