import pickle
import matplotlib.pyplot as plt
import umap

def plot_embedding():
    with open("data/embedding_20k.pkl", "rb") as fp:
        corpus = pickle.load(fp)
    reducer = umap.UMAP(n_components=2, random_state=0)
    
    skip_size = 1 # 重い場合
    normed_embedding = reducer.fit_transform(corpus["original_embedding"][::skip_size])
    vocab = corpus["word"][::skip_size]

    # 次元削減の結果を保存
    with open("data/umap_reducer.pkl", "wb") as fp:
        pickle.dump(reducer, fp)

    fig = plt.figure(figsize=(10, 10))
    for i in range(normed_embedding.shape[0]):
        plt.text(normed_embedding[i, 0], normed_embedding[i, 1], vocab[i], ha="center", va="center", c="black")
    plt.xlim((normed_embedding[:, 0].min(), normed_embedding[:, 0].max()))
    plt.ylim((normed_embedding[:, 1].min(), normed_embedding[:, 1].max()))
    plt.show()

if __name__ == "__main__":
    plot_embedding()