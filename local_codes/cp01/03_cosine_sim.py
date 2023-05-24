import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import cosine_similarity

def cosine_manual():
    iris = load_iris()
    _, n_col = iris["data"].shape
    cosine_sim = np.zeros((n_col, n_col))
    for i in range(n_col):
        for j in range(n_col):
            x, y = iris["data"][:, i], iris["data"][:, j]
            cosine_sim[i, j] = (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))
    print(cosine_sim)
    # [[1.         0.9780132  0.94845135 0.89769132]
    #  [0.9780132  1.         0.87109737 0.80882106]
    #  [0.94845135 0.87109737 1.         0.98354968]
    #  [0.89769132 0.80882106 0.98354968 1.        ]]

def sklearn_cosine():
    iris = load_iris()
    cosine_sim = cosine_similarity(iris["data"].T)
    print(cosine_sim)
    # [[1.         0.9780132  0.94845135 0.89769132]
    #  [0.9780132  1.         0.87109737 0.80882106]
    #  [0.94845135 0.87109737 1.         0.98354968]
    #  [0.89769132 0.80882106 0.98354968 1.        ]]

def gram_matrix_cosine():
    iris = load_iris()
    z = (iris["data"] / np.linalg.norm(iris["data"], axis=0, keepdims=True)).T
    gram_matrix = z @ z.T
    print(gram_matrix)
    # [[1.         0.9780132  0.94845135 0.89769132]
    #  [0.9780132  1.         0.87109737 0.80882106]
    #  [0.94845135 0.87109737 1.         0.98354968]
    #  [0.89769132 0.80882106 0.98354968 1.        ]]