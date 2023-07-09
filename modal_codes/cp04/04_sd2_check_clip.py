from transformers import AutoTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline
import torch
import modal

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

def inference_clip(word_list, model, tokenizer, device):
    with torch.no_grad():
        inputs = tokenizer(word_list, padding=True, return_tensors="pt").to(device)

        outputs = model(**inputs)
        return outputs.last_hidden_state, outputs.pooler_output

def check_embedding_sd2():
    with open("data/corpus_20k.txt", "r", encoding="utf-8") as fp:
        word_list = fp.read().split("\n")

    device = "cuda"
    clip_model = CLIPTextModel.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=CACHE_DIR).to(device)
    clip_tokenizer = AutoTokenizer.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=CACHE_DIR)
    clip_last_hidden_state, clip_pooled_output = inference_clip(word_list, clip_model, clip_tokenizer, device)

    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", cache_dir=CACHE_DIR)
    sd_text_model = sd_pipe.text_encoder.to(device)
    sd_tokenizer = sd_pipe.tokenizer
    sd_last_hidden_state, sd_pooled_output = inference_clip(word_list, sd_text_model, sd_tokenizer, device)

    diff_hidden = torch.sum(torch.abs(clip_last_hidden_state - sd_last_hidden_state)).cpu().numpy()
    diff_pooled = torch.sum(torch.abs(clip_pooled_output - sd_pooled_output)).cpu().numpy()
    print(diff_hidden, diff_pooled) # 31428138.0 4070389.0
    # オリジナルCLIPと、SDのCLIPではレイヤー数が異なる
    print(len(clip_model.text_model.encoder.layers), len(sd_text_model.text_model.encoder.layers)) # 24 23

def check_embedding_sd2_delete_last():
    with open("data/corpus_20k.txt", "r", encoding="utf-8") as fp:
        word_list = fp.read().split("\n")

    device = "cuda"
    clip_model = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=CACHE_DIR)
    clip_tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=CACHE_DIR)
    # 最終層を削除
    last_layer = clip_model.text_model.encoder.layers.pop(-1)
    clip_model = clip_model.to(device)
    clip_last_hidden_state, clip_pooled_output = inference_clip(word_list, clip_model, clip_tokenizer, device)

    sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", cache_dir=CACHE_DIR)
    sd_text_model = sd_pipe.text_encoder.to(device)
    sd_tokenizer = sd_pipe.tokenizer
    sd_last_hidden_state, sd_pooled_output = inference_clip(word_list, sd_text_model, sd_tokenizer, device)

    diff_hidden = torch.sum(torch.abs(clip_last_hidden_state - sd_last_hidden_state)).cpu().numpy()
    diff_pooled = torch.sum(torch.abs(clip_pooled_output - sd_pooled_output)).cpu().numpy()
    print(diff_hidden, diff_pooled) # 20215248.0 0.0
    print(len(clip_model.text_model.encoder.layers), len(sd_text_model.text_model.encoder.layers)) # 23 23

@stub.function(
    image=modal.Image.debian_slim().pip_install("torch", "diffusers[torch]", "transformers"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def main():
    print("--check_embedding_sd2--")
    check_embedding_sd2()
    print("--check_embedding_sd2_delete_last--")
    check_embedding_sd2_delete_last()

if __name__ == "__main__":
    with stub.run():
        main.call()