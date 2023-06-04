from PIL import Image
import torch
from transformers import AutoProcessor, CLIPModel
import numpy as np
from diffusers import StableDiffusionPipeline

def fine_tuned_image_text_retrival(img_path,
                                   sd_model_names,
                                   additional_corpus_words=None,
                                   device="cuda"):
    with open("data/corpus_20k.txt", "r", encoding="utf-8") as fp:
        word_list = fp.read().split("\n")
    if additional_corpus_words is not None:
        word_list += additional_corpus_words

    original_clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    original_clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    for sd_model_name in sd_model_names:
        pipe = StableDiffusionPipeline.from_pretrained(sd_model_name)
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

def infernce_finetuned_pikachu_images():
    images = [
        "data/06_1_pikachu.png",
        "data/06_2_pikachu.png",
        "data/06_3_pikachu.png",
        "data/06_4_free_pikachu.jpg",
    ]
    for img_path in images:
        print("----", img_path, "----")
        fine_tuned_image_text_retrival(img_path,
                                    ["runwayml/stable-diffusion-v1-5",
                                    "Envvi/Inkpunk-Diffusion",
                                    "nitrosocke/redshift-diffusion"], ["nvinkpunk", "redshift style"])     

if __name__ == "__main__":
    infernce_finetuned_pikachu_images()