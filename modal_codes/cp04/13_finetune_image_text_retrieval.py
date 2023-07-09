from PIL import Image
import torch
from transformers import AutoProcessor, CLIPModel
import numpy as np
from diffusers import StableDiffusionPipeline
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
def fine_tuned_image_text_retrival(img_path,
                                   sd_model_names,
                                   additional_corpus_words=None,
                                   device="cuda"):
    with open("data/corpus_20k.txt", "r", encoding="utf-8") as fp:
        word_list = fp.read().split("\n")
    if additional_corpus_words is not None:
        word_list += additional_corpus_words

    original_clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14", cache_dir=CACHE_DIR).to(device)
    original_clip_processor = AutoProcessor.from_pretrained(
        "openai/clip-vit-large-patch14", cache_dir=CACHE_DIR)

    for sd_model_name in sd_model_names:
        pipe = StableDiffusionPipeline.from_pretrained(sd_model_name, cache_dir=CACHE_DIR)
        text_model = pipe.text_encoder.to(device)
        tokenizer = pipe.tokenizer

        with torch.no_grad():
            inputs = tokenizer(word_list, padding=True, return_tensors="pt").to(device)
            outputs = text_model(**inputs)
            text_embeds = original_clip_model.text_projection(outputs.pooler_output)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds.cpu().numpy()

            inputs = original_clip_processor(images=Image.open(img_path), return_tensors="pt").to(device)
            outputs = original_clip_model.vision_model(**inputs)       
            image_embeds = original_clip_model.visual_projection(outputs.pooler_output)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            image_embeds = image_embeds.cpu().numpy()

        similarity = (image_embeds @ text_embeds.T)[0]
        max_indices = np.argsort(similarity)[::-1][:10]

        print("\n---", sd_model_name, "--")
        print(np.array(word_list)[max_indices].tolist())
        print((similarity[max_indices]*100).astype(np.int32).tolist())

if __name__ == "__main__":
    with stub.run():
        fine_tuned_image_text_retrival.call(
            "data/01_base.png", [
                "runwayml/stable-diffusion-v1-5",
                "prompthero/openjourney-v4",
                "NoCrypt/SomethingV2_2"])