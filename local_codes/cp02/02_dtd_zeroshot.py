import torch
import torch.utils.data
import torchvision
from tqdm import tqdm
import numpy as np
import os
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, AutoProcessor

def get_prompt_embedding():
    if not os.path.exists("output/07/class_embedding.pt"):
        clip_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")    
        
        dataset = torchvision.datasets.DTD(root="./dataset", download=True, split="test")
        prompt_templates = [
            'a photo of a {} texture.',
            'a photo of a {} pattern.',
            'a photo of a {} thing.',
            'a photo of a {} object.',
            'a photo of the {} texture.',
            'a photo of the {} pattern.',
            'a photo of the {} thing.',
            'a photo of the {} object.',
        ]
        list_of_classes = list(dataset.class_to_idx.keys())
        class_text_embedding = []
        for c in tqdm(list_of_classes):
            text = [f.format(c) for f in prompt_templates]
            with torch.no_grad():
                token_indices = clip_processor(text, return_tensors="pt", padding=True)
                text_outputs = clip_model(**token_indices)
                text_features = text_outputs["text_embeds"] / text_outputs["text_embeds"].norm(p=2, dim=-1, keepdim=True)
            class_text_embedding.append(text_features.mean(dim=0, keepdim=True))

        class_text_embedding = torch.cat(class_text_embedding, dim=0)
        class_text_embedding /= class_text_embedding.norm(dim=-1, keepdim=True)
        result = {
            "class_names": dataset.class_to_idx,
            "embeddings": class_text_embedding.cpu()
        }    
        torch.save(result, "output/07/class_embedding.pt")
    else:
        result = torch.load("output/07/class_embedding.pt")
    
    return result

def get_image_embedding(split):
    filepath = f"output/07/image_{split}.pt"
    if not os.path.exists(filepath):
        clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")    
    
        dataset = torchvision.datasets.DTD(root="./dataset", download=True, split=split)
        image_embddings, class_indices = [], []
        for item in tqdm(dataset):
            img, class_id = item
            with torch.no_grad():
                inputs = clip_processor(images=img, return_tensors="pt")
                image_outputs = clip_model(**inputs)
                image_features = image_outputs["image_embeds"] / image_outputs["image_embeds"].norm(p=2, dim=-1, keepdim=True)
            image_embddings.append(image_features)
            class_indices.append(class_id)
        image_embddings = torch.cat(image_embddings, dim=0)
        result = {
            "class_idx": class_indices,
            "embeddings": image_embddings
        }
        torch.save(result, filepath)
    else:
        result = torch.load(filepath)
    return result

def eval_zeroshot():
    text_data = get_prompt_embedding()
    image_data = get_image_embedding("test")
    with torch.no_grad(), torch.cuda.amp.autocast():
        affine = image_data["embeddings"] @ text_data["embeddings"].T
        y_pred = affine.argmax(dim=-1).cpu().numpy()
        y_true = np.array(image_data["class_idx"])

    correct_flags = np.array(y_true == y_pred)
    print(correct_flags.mean()) # 0.4420212765957447