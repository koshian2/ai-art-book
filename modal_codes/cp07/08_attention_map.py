import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch
import modal
import matplotlib.pyplot as plt
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

class StoreAttentionMapProcessor:
    def __init__(self, module_name, width, height):
        super().__init__()
        self.module_name = module_name
        self.cross_attention_history = []
        self.width = width
        self.height = height
        self.device = "cuda"

    def try_get_attention_map(self, get_index, is_positive=True, min_rate_threshold=0):
        if len(self.cross_attention_history) == 0:
            return None
        attn_map = self.cross_attention_history[get_index].mean(dim=1) # (2, dim, n_token)
        attn_map = attn_map[1] if is_positive else attn_map[0] # (dim, n_token)
        rate = int((self.height * self.width // attn_map.shape[0]) ** 0.5)
        if rate <= min_rate_threshold:
            return None
        down_height = self.height // rate
        down_width = self.width // rate
        attn_map = attn_map.view(down_height, down_width, attn_map.shape[1]).permute(2, 0, 1).unsqueeze(0) # (1, n_toke, H, W)
        attn_map = attn_map.to(self.device)
        attn_map = F.interpolate(attn_map, (self.height, self.width), mode="bicubic").squeeze(0) # (n_token, H, W)
        return attn_map

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        is_cross_attn = encoder_hidden_states is not None
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if is_cross_attn:
            store_attention = attention_probs.view(hidden_states.shape[0], -1, attention_probs.shape[1], attention_probs.shape[2])
            self.cross_attention_history.append(store_attention.detach().cpu())
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    

def list_attention_processors(pipe):
    def recursive_list_attention_processor(name, module):
        if hasattr(module, "set_processor"):
            yield name, module
        else:
            for sub_name, child in module.named_children():
                yield from recursive_list_attention_processor(f"{name}.{sub_name}", child)

    for name, module in pipe.unet.named_children():
        yield from recursive_list_attention_processor(name, module)

def get_indices(tokenizer, prompt):
    """Utility function to list the indices of the tokens you wish to alte"""
    ids = tokenizer(prompt).input_ids
    indices = {i: tok for tok, i in zip(tokenizer.convert_ids_to_tokens(ids), range(len(ids)))}
    return indices

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "transformers", "matplotlib"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def main(width=960, height=512):
    device = "cuda"    
    pipe = StableDiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2", torch_dtype=torch.float16, cache_dir=CACHE_DIR)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    for name, module in list_attention_processors(pipe):
        module.set_processor(StoreAttentionMapProcessor(name, width, height))

    pipe.enable_vae_tiling()
    pipe.to(device)

    prompt = "two girls standing in a lavender field in the countryside and having fun, best quality, extremely detailed"
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    token_indices = get_indices(pipe.tokenizer, prompt)

    generator = torch.Generator(device).manual_seed(1234)
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, width=width, height=height,
                 generator=generator, num_inference_steps=30).images[0]

    attn_maps = []
    for name, module in list_attention_processors(pipe):
        attn_map = module.processor.try_get_attention_map(-1, min_rate_threshold=8)
        if attn_map is not None:
            attn_maps.append(attn_map)
    attn_maps = torch.stack(attn_maps, dim=0).mean(dim=0).cpu().numpy()

    image.save(f"{CACHE_DIR}/output/08_attention_map_original_image.jpg", quality=92)

    fig = plt.figure(figsize=(20, 8))
    for i in range(18):
        ax = fig.add_subplot(3, 6, i+1)
        ax.imshow(attn_maps[i])
        ax.axis("off")
        ax.set_title(token_indices[i])
    fig.savefig(f"{CACHE_DIR}/output/08_attention_map.png")

if __name__ == "__main__":
    with stub.run():
        main.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/08_attention_map* .', shell=True)
