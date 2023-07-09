from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
import torch.nn.functional as F
import matplotlib.pyplot as plt
import modal
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

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
            query_uncond, query_cond = query.chunk(2)
            queries = [query_uncond for i in range(self.mask.shape[0])]
            queries += [query_cond for i in range(self.mask.shape[0])]
            query = torch.cat(queries, dim=0)

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

            uncond, cond = hidden_states.chunk(2) # (N, H, W, C)
            spatial_mask = F.interpolate(self.mask, (height, width), mode="nearest")
            spatial_mask = spatial_mask.permute(0, 2, 3, 1) # (N, 1, H, W) -> (N, H, W, 1)
            spatial_mask = spatial_mask.reshape(spatial_mask.shape[0], -1, spatial_mask.shape[3]) # (N, HW, 1)
            mask_normed =  spatial_mask / (torch.sum(spatial_mask, dim=0, keepdims=True) + 1e-5)
            masked_cond = torch.sum(cond * mask_normed, dim=0, keepdim=True)
            masked_uncond = torch.sum(uncond * mask_normed, dim=0, keepdim=True)
            hidden_states = torch.cat([masked_uncond, masked_cond])

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

class AttentionCouplePipeline(StableDiffusionPipeline):    
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

def compare_explorer(width=960*2, height=512*2):
    result = []
    device = "cuda"    
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    pipe.to(device)

    ## 1. Normal Run
    prompt = "a girl adventurer is walking alone on a hill by the sea, mountains can be seen in the distance in the background, a small field of flowers, 1girl, blue shirts, green vest, knee-length skirt, blonded-hair, leather hair bands, masterpiece, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    generator = torch.Generator().manual_seed(1234)
    latent = torch.randn((1, 4, height//8, width//8), generator=generator).to(device, torch.float16)
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=12,
                 latents=latent, num_inference_steps=50).images[0]
    result.append(image)

    ## 2. Attention Couple + Different Neg Prompt
    pipe = AttentionCouplePipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()

    prompts = [
        "a girl adventurer is walking alone on a hill by the sea, mountains can be seen in the distance in the background, masterpiece, best quality, extremely detailed",
        "a girl adventurer is walking alone on a hill by the sea, 1girl, blue shirts, green vest, knee-length skirt, blonded-hair, leather hair bands, masterpiece, best quality, extremely detailed",
        "a hill by the sea, mountains can be seen in the distance in the background, a small field of flowers, masterpiece, best quality, extremely detailed"
    ]
    negative_prompts = [
        "bad anatomy, cropped, worst quality, low quality",
        "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, extra legs, cropped, worst quality, low quality",
        "bad anatomy, cropped, worst quality, low quality",
    ]
    # 全体、左1/3、右でマスクを作る
    all_mask = torch.ones((1, 1, height//8, width//8), dtype=torch.float16)
    left_mask, right_mask = all_mask.clone(), all_mask.clone()
    left_mask[:, :, :, width//24:] = 0 # 横1/3～をオフ
    right_mask[:, :, :, :width//24] = 0 # ～横1/3をオフ
    masks = torch.cat([all_mask, left_mask*3.5, right_mask], dim=0).to(device)

    pipe.enable_attention_couple(width, height, masks)
    pipe.to(device)

    # 乱数は1個で初期化
    generator = torch.Generator().manual_seed(1234)
    latent = torch.randn((1, 4, height//8, width//8), generator=generator).to(device, torch.float16)
    image = pipe(prompt=prompts, negative_prompt=negative_prompts, guidance_scale=12,
                 latents=latent, num_inference_steps=50).images[0]
    result.append(image)

    ## 3. Attention Couple + Same Neg Prompt
    prompts = [
        "a girl adventurer is walking alone on a hill by the sea, mountains can be seen in the distance in the background, masterpiece, best quality, extremely detailed",
        "a girl adventurer is walking alone on a hill by the sea, 1girl, blue shirts, green vest, knee-length skirt, blonded-hair, leather hair bands, masterpiece, best quality, extremely detailed",
        "a hill by the sea, mountains can be seen in the distance in the background, a small field of flowers, masterpiece, best quality, extremely detailed"
    ]
    negative_prompts = [
        "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, extra legs, cropped, worst quality, low quality",
        "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, extra legs, cropped, worst quality, low quality",
        "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, extra legs, cropped, worst quality, low quality",
    ]
    image = pipe(prompt=prompts, negative_prompt=negative_prompts, guidance_scale=12,
                 latents=latent, num_inference_steps=50).images[0]
    result.append(image)

    return result

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]==0.16.1", "transformers", "matplotlib"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    timeout=600,
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def main():
    images = compare_explorer()
    titles = ["normal", "different_neg_prompt", "same_neg_prompt"]
    fig = plt.figure(figsize=(20, 8))
    for i, title in enumerate(titles):
        ax = fig.add_subplot(1, 3, i+1)
        ax.imshow(images[i])
        ax.axis("off")
        ax.set_title(title)
        images[i].save(f"{CACHE_DIR}/output/02_attn_couple_{i+1}_{title}.jpg", quality=92)
    fig.savefig(f"{CACHE_DIR}/output/02_attn_couple_compare.png")

if __name__ == "__main__":
    with stub.run():
        main.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/02_attn_couple* .', shell=True)
