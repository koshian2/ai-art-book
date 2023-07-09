import torch
from diffusers import DiffusionPipeline
from transformers import AutoProcessor, CLIPModel
import numpy as np
from PIL import Image
from utils import load_safetensors_lora
from settings import LORA_DIRECTORY
import pickle
import modal
import subprocess

CACHE_DIR = "/cache"
volume = modal.NetworkFileSystem.persisted("model-cache-vol")

stub = modal.Stub()

def fine_tuned_image_text_retrival(img_paths,
                                   pipe,
                                   additional_corpus_words=None,
                                   device="cuda"):
    with open("data/embedding_20k.pkl", "rb") as fp:
        data = pickle.load(fp)
    word_list = data["word"]
    if additional_corpus_words is not None:
        word_list += additional_corpus_words

    original_clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=CACHE_DIR).to(device)
    original_clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=CACHE_DIR)

    text_model = pipe.text_encoder.to(device)
    tokenizer = pipe.tokenizer

    with torch.no_grad():
        inputs = tokenizer(word_list, padding=True, return_tensors="pt").to(device)
        outputs = text_model(**inputs)
        text_embeds = original_clip_model.text_projection(outputs.pooler_output)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds.cpu().numpy()

        images = [Image.open(x) for x in img_paths]
        inputs = original_clip_processor(images=images, return_tensors="pt").to(device)
        outputs = original_clip_model.vision_model(**inputs)       
        image_embeds = original_clip_model.visual_projection(outputs.pooler_output)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        image_embeds = image_embeds.cpu().numpy()

    similarity_matrix = image_embeds @ text_embeds.T
    for i in range(similarity_matrix.shape[0]):
        sim = similarity_matrix[i]
        max_indices = np.argsort(sim)[::-1][:10]

        print(np.array(word_list)[max_indices].tolist())
        print((sim[max_indices]*100).astype(np.int32).tolist())

@stub.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "diffusers[torch]", "transformers"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    gpu="t4",
    network_file_systems={CACHE_DIR: volume},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root")]
)
def check_text_retrieval(enable_lora):
    pipe = DiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2",
         custom_pipeline="lpw_stable_diffusion", cache_dir=CACHE_DIR)
    if enable_lora:
        pipe = load_safetensors_lora(pipe, f"{LORA_DIRECTORY}/lumine1-000008.safetensors", alpha=0.4)
    fine_tuned_image_text_retrival(
        ["data/lumine_01.png", "data/lumine_02.png", "data/lumine_03.png",
         "data/non_lumine_01.jpg", "data/non_lumine_02.jpg", "data/non_lumine_03.jpg"],
        pipe, ["luminedef", "luminernd"]
    )

def main():
    with stub.run():
        print("---Disable LoRA--")
        check_text_retrieval.call(False)
        print("---Enable LoRA--")
        check_text_retrieval.call(True)

if __name__ == "__main__":
    main()