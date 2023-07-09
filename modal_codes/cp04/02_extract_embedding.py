from transformers import AutoTokenizer, CLIPTextModel
import torch
import modal

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
def extract_text_embedding():
    with open("data/corpus_20k.txt", "r", encoding="utf-8") as fp:
        word_list = fp.read().split("\n")

    device = "cuda"
    clip_model = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14", cache_dir=CACHE_DIR).to(device)
    clip_tokenizer = AutoTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14", cache_dir=CACHE_DIR)

    with torch.no_grad():
        inputs = clip_tokenizer(word_list, padding=True, return_tensors="pt").to(device)

        outputs = clip_model(**inputs)
        clip_last_hidden_state = outputs.last_hidden_state
        clip_pooled_output = outputs.pooler_output  # pooled (EOS token) states
    print(clip_last_hidden_state.shape) # Size([7582, 7, 768])
    print(clip_pooled_output.shape) # torch.Size([7582, 768])

if __name__ == "__main__":
    with stub.run():
        extract_text_embedding.call()