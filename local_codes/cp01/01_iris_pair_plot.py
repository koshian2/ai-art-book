from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def pair_plot():
    iris = load_iris()
    df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
    df['class'] = iris.target_names[iris.target]
    sns.pairplot(data=df, hue='class', diag_kind='hist')
    plt.show()

if __name__ == "__main__":
    pair_plot()