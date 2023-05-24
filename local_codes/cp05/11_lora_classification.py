import torch
from diffusers import DiffusionPipeline
from transformers import AutoProcessor, CLIPModel
from safetensors.torch import load_file
import numpy as np
from PIL import Image

def load_safetensors_lora(pipeline,
                          checkpoint_path,
                          LORA_PREFIX_UNET = "lora_unet",
                          LORA_PREFIX_TEXT_ENCODER = "lora_te",
                          alpha = 0.75):
    # 先程と同じなので省略


def fine_tuned_image_text_retrival(img_paths,
                                   pipe,
                                   additional_corpus_words=None,
                                   device="cuda:1"):
    with open("data/corpus_20k.txt", "r", encoding="utf-8") as fp:
        word_list = fp.read().split("\n")
    if additional_corpus_words is not None:
        word_list += additional_corpus_words

    original_clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    original_clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

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

def main(enable_lora):
    pipe = DiffusionPipeline.from_pretrained(
        "NoCrypt/SomethingV2_2",
         custom_pipeline="lpw_stable_diffusion")
    if enable_lora:
        pipe = load_safetensors_lora(pipe, "H:/diffusion_models/lora/lumine1-000008.safetensors", alpha=0.4)
    fine_tuned_image_text_retrival(
        ["data/lumine_01.png", "data/lumine_02.png", "data/lumine_03.png",
         "data/non_lumine_01.jpg", "data/non_lumine_02.jpg", "data/non_lumine_03.jpg"],
        pipe, ["luminedef", "luminernd"]
    )

if __name__ == "__main__":
    main(False)
