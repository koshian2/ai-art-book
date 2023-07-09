from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import modal

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

def calc_metrics(embeddings, key, mode):
    results = []
    for i in range(len(embeddings)):
        row = []
        for j in range(len(embeddings)):
            x1 = embeddings[i][key]
            x2 = embeddings[j][key]
            if x1.ndim == 3:
                x1 = x1.reshape(x1.shape[0], -1)
                x2 = x2.reshape(x2.shape[0], -1)
            if mode == "mae":
                row.append(torch.sum(torch.abs(x1 - x2), dim=-1).mean().cpu().numpy())
            elif mode == "cosine":
                xn1 = x1 / x1.norm(p=2, dim=-1, keepdim=True)
                xn2 = x2 / x2.norm(p=2, dim=-1, keepdim=True)
                row.append(torch.sum(xn1 * xn2, dim=-1).mean().cpu().numpy())
        results.append(row)
    return np.array(results)

@stub.function(
    image=modal.Image.debian_slim().pip_install("torch", "diffusers[torch]", "transformers"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def compare_finetuned_sd():
    with open("data/corpus_20k.txt", "r", encoding="utf-8") as fp:
        word_list = fp.read().split("\n")

    device = "cuda"
    model_names = [
        "runwayml/stable-diffusion-v1-5",
        "prompthero/openjourney-v4",
        "NoCrypt/SomethingV2_2"
    ]
    embeddings = []
    for model_name in model_names:
        pipe = StableDiffusionPipeline.from_pretrained(model_name, cache_dir=CACHE_DIR)
        text_model = pipe.text_encoder.to(device)
        tokenizer = pipe.tokenizer
        with torch.no_grad():
            inputs = tokenizer(word_list, padding=True, return_tensors="pt").to(device)
            outputs = text_model(**inputs)
            item = {
                "model_name": model_name,
                "last_hidden_state": outputs.last_hidden_state,
                "pooler_output": outputs.pooler_output
            }
            embeddings.append(item)

    results = []
    results.append(calc_metrics(embeddings, "last_hidden_state", "mae"))
    results.append(calc_metrics(embeddings, "pooler_output", "mae"))
    results.append(calc_metrics(embeddings, "last_hidden_state", "cosine"))
    results.append(calc_metrics(embeddings, "pooler_output", "cosine"))
    results = np.concatenate(results, axis=0)
    print(results)

if __name__ == "__main__":
    with stub.run():
        compare_finetuned_sd.call()