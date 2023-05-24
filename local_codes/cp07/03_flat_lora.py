from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch
from safetensors.torch import load_file
import matplotlib.pyplot as plt

def load_safetensors_lora(pipeline,
                          checkpoint_path,
                          LORA_PREFIX_UNET = "lora_unet",
                          LORA_PREFIX_TEXT_ENCODER = "lora_te",
                          alpha = 0.75):
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path)

    visited = []
    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))
        # print(curr_layer, pair_keys)

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)

    return pipeline


def run_flat_lora(prompt,
                  model_name="stabilityai/stable-diffusion-2-1-base",
                  device="cuda", seed=1234):
    fig = plt.figure(figsize=(18, 10))
    for i, lora_name in enumerate(["bigeye.safetensors", "flat2.safetensors"]):
        alphas = [-1, -0.5, 0, 0.5, 1]
        for j, alpha in enumerate(alphas):
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name, torch_dtype=torch.float16)
            pipe = load_safetensors_lora(pipe, f"H:/diffusion_models/lora/{lora_name}", alpha=alpha)

            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.safety_checker=lambda images, **kwargs: (images, False)
            pipe.vae.enable_tiling()
            pipe.to(device)

            negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

            generator = torch.Generator(device).manual_seed(seed)
            image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=30, 
                        generator=generator, height=768).images[0]
            
            ax = fig.add_subplot(2, 5, i*(len(alphas))+j+1)
            ax.imshow(image)
            ax.set_title(f"{lora_name} | α={alpha}")
            ax.axis("off")

    plt.imshow(image)
    plt.savefig("output/07/02_style_lora.png")
    plt.show()

if __name__ == "__main__":
    run_flat_lora("1girl, look at viewer, best quality", model_name="H:/diffusion_models/diffusers/AnythingV5_v5PrtRE")
