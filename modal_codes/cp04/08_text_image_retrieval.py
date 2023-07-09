from PIL import Image
import torch
import pickle
from transformers import AutoProcessor, CLIPVisionModelWithProjection
import numpy as np
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
def text_image_retrieval(img_path):
    with open("data/embedding_20k.pkl", "rb") as fp:
        data = pickle.load(fp)

    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", cache_dir=CACHE_DIR)
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=CACHE_DIR)

    with torch.no_grad():
        inputs = clip_processor(images=Image.open(img_path), return_tensors="pt")
        outputs = clip_model(**inputs)        
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        image_embeds = image_embeds.numpy()

    similarity = (image_embeds @ data["embedding"].T)[0]
    print(similarity.shape)
    max_indices = np.argsort(similarity)[::-1][:10]

    print("\n---")
    print(np.array(data["word"])[max_indices].tolist())
    print((similarity[max_indices]*100).astype(np.int32).tolist())

if __name__ == "__main__":
    with stub.run():
        text_image_retrieval.call("data/01_base.png")