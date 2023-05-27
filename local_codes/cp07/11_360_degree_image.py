import torch
from diffusers import UniPCMultistepScheduler, StableDiffusionPipeline
from safetensors.torch import load_file
import copy

def load_multiple_safetensors_lora(pipeline,
                                   checkpoint_paths,
                                   alphas,
                                   LORA_PREFIX_UNET = "lora_unet",
                                   LORA_PREFIX_TEXT_ENCODER = "lora_te"):
    summaries = {}
    for checkpoint_path, alpha in zip(checkpoint_paths, alphas):
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
                    prev_layer = curr_layer
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

            if curr_layer not in summaries.keys():
                summaries[curr_layer] = [] 
            summaries[curr_layer].append([
                prev_layer,
                curr_layer,
                temp_name,
                [state_dict[pair_keys[0]], state_dict[pair_keys[1]]],
                alpha, 
                pair_keys
            ])

            # update visited list
            for item in pair_keys:
                visited.append(item)
        
    # assign joint layer
    for key, values in summaries.items():
        lora_layers = [lora_item[3] for lora_item in values]
        alphas = [lora_item[4] for lora_item in values]
        prev_layer, curr_layer, temp_name, _, alpha, pair_keys = values[0]
        layer = copy.deepcopy(curr_layer)
        for (weight_up, weight_down), alpha in zip(lora_layers, alphas):
            if len(weight_up.shape) == 4:
                weight_up = weight_up.squeeze(3).squeeze(2).to(torch.float32)
                weight_down = weight_down.squeeze(3).squeeze(2).to(torch.float32)
                layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                weight_up = weight_up.to(torch.float32)
                weight_down = weight_down.to(torch.float32)
                layer.weight.data += alpha * torch.mm(weight_up, weight_down)        
        prev_layer.__setattr__(temp_name, layer)

    return pipeline

import matplotlib.pyplot as plt
from omnidirectional_viewer import omniview

def main():
    device = "cuda:1"
    model_id = "runwayml/stable-diffusion-v1-5"
    model_id = "NoCrypt/SomethingV2_2"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16)
    pipe.enable_vae_tiling()
    pipe = load_multiple_safetensors_lora(
        pipe,
        [
            "data/LatentLabs360.safetensors",
        ],
        [1.0], "lora_unet", "lora_te"
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device, torch.float16)

    generator = torch.Generator().manual_seed(1234)
    prompt = "a 360 equirectangular panorama, modelshoot style, bedroom in a rich mansion, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, 
                 generator=generator, width=1024, height=512,
                 num_inference_steps=30).images[0]
    image.save("output/09/11_360dregree.png")
    omniview("output/09/11_360dregree.png", width=640*2, height=360*2)

if __name__ == "__main__":
    main()
    # visualize()
