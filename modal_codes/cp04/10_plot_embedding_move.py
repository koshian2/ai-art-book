from transformers import AutoTokenizer,  CLIPTextModelWithProjection
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

def embedding_move(queries):
    with open("data/embedding_20k.pkl", "rb") as fp:
        corpus = pickle.load(fp)

    # 次元削減の結果を保存
    with open("data/umap_reducer.pkl", "rb") as fp:
        reducer =  pickle.load(fp)

    skip_size = 10
    normed_embedding = reducer.transform(corpus["original_embedding"][::skip_size])
    vocab = corpus["word"][::skip_size]

    clip_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    with torch.no_grad():
        inputs = clip_tokenizer(queries, padding=True, return_tensors="pt")
        outputs = clip_model(**inputs)
        query_embedding = outputs.text_embeds.cpu().numpy()
    
    normed_queries = reducer.transform(query_embedding)

    corrcoef = np.corrcoef(query_embedding, rowvar=True)
    print()
    print((corrcoef*100).astype(np.int32))

    for i in range(normed_embedding.shape[0]):
        plt.text(normed_embedding[i, 0], normed_embedding[i, 1], vocab[i], ha="center", va="center", c="black")
    for i in range(normed_queries.shape[0]):
        plt.annotate(str(i), (normed_queries[i,0], normed_queries[i, 1]), c="red", fontsize=20)
    plt.plot(normed_queries[:, 0], normed_queries[:, 1], c="red", marker=".")

    plt.xlim((normed_embedding[:, 0].min(), normed_embedding[:, 0].max()))
    plt.ylim((normed_embedding[:, 1].min(), normed_embedding[:, 1].max()))
    plt.show()

def main():
    prompt = "a girl adventurer is walking alone on a hill by the sea, mountains can be seen in the distance in the background, a small field of flowers, 1girl, blue shirts, green vest, knee-length skirt, blonded-hair, leather hair bands, masterpiece, best quality, extremely detailed"
    prompts = prompt.split(", ")
    embedding_move(prompts)

if __name__ == "__main__":
    main()
