from PIL import Image
import torch
import pickle
from transformers import AutoProcessor, CLIPModel
import numpy as np
from diffusers import StableDiffusionPipeline

def fine_tuned_text_text_retrival(text_query,
                                  sd_model_names,
                                  additional_corpus_words=None,
                                  device="cuda:1"):
    with open("data/corpus_20k.txt", "r", encoding="utf-8") as fp:
        word_list = fp.read().split("\n")
    if additional_corpus_words is not None:
        word_list += additional_corpus_words

    for sd_model_name in sd_model_names:
        pipe = StableDiffusionPipeline.from_pretrained(sd_model_name)
        text_model = pipe.text_encoder.to(device)
        tokenizer = pipe.tokenizer
        with torch.no_grad():
            inputs = tokenizer(word_list + [text_query], padding=True, return_tensors="pt").to(device)
            outputs = text_model(**inputs)
            text_embeds = outputs.pooler_output
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds.cpu().numpy()
            query_embeds, text_embeds = text_embeds[-1:], text_embeds[:-1]

        similarity = (query_embeds @ text_embeds.T)[0]
        max_indices = np.argsort(similarity)[::-1][:10]

        print("\n---", text_query, sd_model_name, "--")
        print(np.array(word_list)[max_indices].tolist())
        print((similarity[max_indices]*100).astype(np.int32).tolist())

if __name__ == "__main__":
    fine_tuned_text_text_retrival("wildlife",
                                  ["runwayml/stable-diffusion-v1-5",
                                   "prompthero/openjourney-v4",
                                   "NoCrypt/SomethingV2_2"])