from transformers import AutoTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline
import torch

def inference_clip(word_list, model, tokenizer, device):
    with torch.no_grad():
        inputs = tokenizer(word_list, padding=True, return_tensors="pt").to(device)

        outputs = model(**inputs)
        return outputs.last_hidden_state, outputs.pooler_output

def check_embedding_sd1():
    with open("data/corpus_20k.txt", "r", encoding="utf-8") as fp:
        word_list = fp.read().split("\n")

    device = "cuda"
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_last_hidden_state, clip_pooled_output = inference_clip(word_list, clip_model, clip_tokenizer, device)

    sd_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    sd_text_model = sd_pipe.text_encoder.to(device)
    sd_tokenizer = sd_pipe.tokenizer
    sd_last_hidden_state, sd_pooled_output = inference_clip(word_list, sd_text_model, sd_tokenizer, device)

    diff_hidden = torch.sum(torch.abs(clip_last_hidden_state - sd_last_hidden_state)).cpu().numpy()
    diff_pooled = torch.sum(torch.abs(clip_pooled_output - sd_pooled_output)).cpu().numpy()
    print(diff_hidden, diff_pooled) # 0.0 0.0

if __name__ == "__main__":
    check_embedding_sd1()