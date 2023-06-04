import numpy as np
from sklearn.datasets import load_iris

def manual():
    iris = load_iris()
    _, n_col = iris["data"].shape
    corrcoef = np.zeros((n_col, n_col))
    for i in range(n_col):
        for j in range(n_col):
            x, y = iris["data"][:, i], iris["data"][:, j]
            a = np.sum((x-np.mean(x)) * (y-np.mean(y)))
            b = np.sqrt(np.sum((x-np.mean(x))**2) * np.sum((y-np.mean(y))**2))
            corrcoef[i, j] = a / b
    print(corrcoef)
    # [[ 1.         -0.11756978  0.87175378  0.81794113]
    #  [-0.11756978  1.         -0.4284401  -0.36612593]
    #  [ 0.87175378 -0.4284401   1.          0.96286543]
    #  [ 0.81794113 -0.36612593  0.96286543  1.        ]]

def calc_corrcoef():
    iris = load_iris()
    corrcoef = np.corrcoef(iris["data"], rowvar=False)
    print(corrcoef)
    # [[ 1.         -0.11756978  0.87175378  0.81794113]
    #  [-0.11756978  1.         -0.4284401  -0.36612593]
    #  [ 0.87175378 -0.4284401   1.          0.96286543]
    #  [ 0.81794113 -0.36612593  0.96286543  1.        ]]

def gram_matrix_corrcoef():
    iris = load_iris()
    mu = np.mean(iris["data"], axis=0, keepdims=True)
    sigma = np.std(iris["data"], axis=0, keepdims=True)
    z = ((iris["data"]-mu)/sigma).T 
    gram_matrix = (z @ z.T) / iris["data"].shape[0]
    print(gram_matrix)
    # [[ 1.         -0.11756978  0.87175378  0.81794113]
    #  [-0.11756978  1.         -0.4284401  -0.36612593]
    #  [ 0.87175378 -0.4284401   1.          0.96286543]
    #  [ 0.81794113 -0.36612593  0.96286543  1.        ]]

def main():
    print("--manual--")
    manual()
    print("--calc_corrcoef--")
    calc_corrcoef()
    print("--gram_matrix_corrcoef--")
    gram_matrix_corrcoef()

if __name__ == "__main__":
    main()