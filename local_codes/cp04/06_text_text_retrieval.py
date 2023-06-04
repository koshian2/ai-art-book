from transformers import AutoTokenizer,  CLIPTextModelWithProjection
import torch
import pickle
import numpy as np

def text_text_retrieval(query_text):
    with open("data/embedding_20k.pkl", "rb") as fp:
        data = pickle.load(fp)

    clip_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    with torch.no_grad():
        inputs = clip_tokenizer([query_text], padding=True, return_tensors="pt")
        outputs = clip_model(**inputs)
        query_embedding = outputs.text_embeds
        query_embedding /= query_embedding.norm(p=2, dim=-1, keepdim=True)
        query_embedding = query_embedding.numpy()

    similarity = (query_embedding @ data["embedding"].T)[0]
    max_indices = np.argsort(similarity)[::-1][:10]

    print("\n---")
    print(np.array(data["word"])[max_indices].tolist())
    print((similarity[max_indices]*100).astype(np.int32).tolist())

if __name__ == "__main__":
    text_text_retrieval("traveler")