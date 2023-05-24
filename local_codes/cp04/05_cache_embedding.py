import pickle

def cache_embedding():
    with open("data/corpus_20k.txt", "r", encoding="utf-8") as fp:
        word_list = fp.read().split("\n")

    device = "cuda:1"
    clip_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    with torch.no_grad():
        inputs = clip_tokenizer(word_list, padding=True, return_tensors="pt").to(device)
        outputs = clip_model(**inputs)
    original_embeds = outputs.text_embeds.cpu().numpy()
    text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds.cpu().numpy()

    with open("data/embedding_20k.pkl", "wb") as fp:
        data = {"word": word_list, "embedding": text_embeds, "original_embedding": original_embeds}
        pickle.dump(data, fp)