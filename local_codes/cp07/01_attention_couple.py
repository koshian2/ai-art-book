from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
import torch.nn.functional as F
from torch import FloatTensor
import os
from PIL import Image
import matplotlib.pyplot as plt

class AttentionCoupleProcessor(AttnProcessor2_0):
    def __init__(self, width, height, region_mask):
        super().__init__()
        self.orig_height = height
        self.orig_width = width
        self.mask = region_mask # (N, C, H, W)

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        inner_dim = hidden_states.shape[-1]
        is_cross_attn = encoder_hidden_states is not None

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        if is_cross_attn:
            # copy query
            query_uncond, query_cond = query.chunk(2)
            queries = [query_cond for i in range(self.mask.shape[0])]
            query = torch.cat([query_uncond] + queries, dim=0)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = inner_dim // attn.heads
        query = query.view(query.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(key.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(value.shape[0], -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(hidden_states.shape[0], -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if is_cross_attn:
            rate = int((self.orig_height * self.orig_width // hidden_states.shape[1]) ** 0.5)
            height = self.orig_height // rate
            width = self.orig_width // rate            

            uncond, cond = hidden_states[0:1], hidden_states[1:] # (N, H, W, C)
            spatial_mask = F.interpolate(self.mask, (height, width), mode="bicubic")
            spatial_mask = spatial_mask.permute(0, 2, 3, 1) # (N, 1, H, W) -> (N, H, W, 1)
            spatial_mask = spatial_mask.reshape(spatial_mask.shape[0], -1, spatial_mask.shape[3]) # (N, HW, 1)
            mask_normed =  spatial_mask / (torch.sum(spatial_mask, dim=0, keepdims=True) + 1e-5)
            masked_cond = torch.sum(cond * mask_normed, dim=0, keepdim=True)
            hidden_states = torch.cat([uncond, masked_cond])

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

class AttentionCouplePipeline(StableDiffusionPipeline):
    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, 
                       negative_prompt=None, prompt_embeds: FloatTensor | None = None, negative_prompt_embeds: FloatTensor | None = None):
        negative_prompt = [negative_prompt for i in range(len(prompt))]
        prompt_embeds = super()._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds, negative_prompt_embeds)
        uncond_prompt, cond_prompt = prompt_embeds.chunk(2)
        prompt_embeds = torch.cat([uncond_prompt[:1], cond_prompt], dim=0)
        print(prompt_embeds.shape)
        return prompt_embeds
    
    def _hack_attention_processor(self, name, module, processors, width, height, region_mask):
        if hasattr(module, "set_processor"):
            processors[f"{name}.processor"] = module.processor
            module.set_processor(AttentionCoupleProcessor(width, height, region_mask))

        for sub_name, child in module.named_children():
            self._hack_attention_processor(f"{name}.{sub_name}", child, processors, width, height, region_mask)

        return processors

    def enable_attention_couple(self, width, height, region_mask):
        processors = {}
        for name, module in self.unet.named_children():
            self._hack_attention_processor(name, module, processors, width, height, region_mask)

def run_attention_couple(prompts, masks, width=960, height=512, device="cuda"):
    pipe = AttentionCouplePipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()

    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    pipe.enable_attention_couple(width, height, masks)
    pipe.to(device)

    # 乱数は1個で初期化
    generator = torch.Generator().manual_seed(1234)
    latent = torch.randn((1, 4, height//8, width//8), generator=generator).to(device, torch.float16)
    image = pipe(prompt=prompts, negative_prompt=negative_prompt, 
                 latents=latent, num_inference_steps=50).images[0]
    return image

def main(width=960, height=512):
    device = "cuda"
    prompts = [
        "two girls standing in a lavender field in the countryside and having fun, best quality, extremely detailed",
        "a girl enjoying the scent of flowers, 1girl, standing, beautiful girl with long blonde hair like a fairy tale princess, blue eyes, white dress, sandals, best quality, extremely detailed",
        "a girl taking a photo, 1girl, standing, healthy girl, wheat-colored skin tan, large eyes, colorful floral shirt, short cut hair, black hair, denim shorts, best quality, extremely detailed",
        "a lavender field in the countryside, best quality, extremely detailed"
    ]
    os.makedirs("output", exist_ok=True)

    # all left right
    all_mask = torch.ones((1, 1, height//8, width//8), dtype=torch.float16)
    left_mask, middle_mask = all_mask.clone(), all_mask.clone()
    left_mask[:, :, :, width//24:] = 0 # 横1/3～をオフ
    middle_mask[:, :, :, :width//24] = 0 # 左1/3をオフ
    middle_mask[:, :, :, width//12:] = 0 # 右1/3をオフ
    masks = torch.cat([all_mask, left_mask*1.5, middle_mask*1.5], dim=0).to(device)
    divide_two = run_attention_couple(prompts[:3], masks, width=width, height=height, device=device)
    divide_two.save("output/01_attention_couple_div2.jpg", quality=92)

   # all left middle right
    all_mask = torch.ones((1, 1, height//8, width//8), dtype=torch.float16)
    left_mask, middle_mask, right_mask = all_mask.clone(), all_mask.clone(), all_mask.clone()
    left_mask[:, :, :, width//24:] = 0 # 横1/3～をオフ
    middle_mask[:, :, :, :width//24] = 0 # 左1/3をオフ
    middle_mask[:, :, :, width//12:] = 0 # 右1/3をオフ
    right_mask[:, :, :, :width//12] = 0 # 横2/3～をオフ
    masks = torch.cat([all_mask, left_mask*1.5, middle_mask*1.5, right_mask], dim=0).to(device)
    divide_three = run_attention_couple(prompts, masks, width=width, height=height, device=device)
    divide_three.save("output/01_attention_couple_div3.jpg", quality=92)

    # visualize
    fig = plt.figure(figsize=(18, 8))
    titles = ["prompt = [all, left, right]", "prompt = [all, left, middle, right]"]
    for i, img in enumerate([divide_two, divide_three]):
        ax = fig.add_subplot(1, 2, i+1)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(titles[i])
    plt.show()

if __name__ == "__main__":
    main()