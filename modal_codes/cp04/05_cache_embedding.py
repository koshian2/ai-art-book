from transformers import AutoTokenizer, CLIPTextModelWithProjection
import torch
import pickle
import modal
import os
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

@stub.function(
    image=modal.Image.debian_slim().pip_install("torch", "diffusers[torch]", "transformers"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def cache_embedding():
    with open("data/corpus_20k.txt", "r", encoding="utf-8") as fp:
        word_list = fp.read().split("\n")

    device = "cuda"
    clip_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", cache_dir=CACHE_DIR).to(device)
    clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14", cache_dir=CACHE_DIR)
    
    with torch.no_grad():
        inputs = clip_tokenizer(word_list, padding=True, return_tensors="pt").to(device)
        outputs = clip_model(**inputs)
    original_embeds = outputs.text_embeds.cpu().numpy()
    text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds.cpu().numpy()

    output_file = f"{CACHE_DIR}/output/embedding_20k.pkl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as fp:
        data = {"word": word_list, "embedding": text_embeds, "original_embedding": original_embeds}
        pickle.dump(data, fp)

if __name__ == "__main__":
    with stub.run():
        cache_embedding.call()
    subprocess.run(
        f'modal nfs get model-cache-vol output/embedding_20k.pkl data/embedding_20k.pkl', shell=True)
