import torch
from diffusers import DiffusionPipeline, UniPCMultistepScheduler
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

def run_lora(width=512, height=960):
    device = "cuda:1"    
    pipe = DiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16,
         custom_pipeline="lpw_stable_diffusion")
    pipe = load_safetensors_lora(pipe, "H:/diffusion_models/lora/lumine1-000008.safetensors", alpha=0.5)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    prompt = "luminedef, luminernd, 1girl, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    generator = torch.Generator(device).manual_seed(1234)
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, max_embeddings_multiples=3,
                 generator=generator, width=width, height=height, num_inference_steps=30).images[0]

    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    run_lora()
