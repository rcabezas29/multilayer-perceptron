from MultilayerPerceptron import MultilayerPerceptron, Layer
import numpy as np
import pandas as pd

def standardize(X) -> np.ndarray:
    """
    Standardize the dataset by removing the mean and scaling to unit variance.
    :param X: The input data.
    :return: The standardized data.
    """
    return (X - X.mean()) / X.std()

df = pd.read_csv("train.csv", header=None)
y = df.iloc[:, 1].apply(lambda x: 1 if x == "M" else 0)
X = df.iloc[:, 2:]
X = standardize(X)

nn = MultilayerPerceptron([
    Layer(X.shape[1], 20, "sigmoid"),
    Layer(20, 10, "sigmoid"),
    Layer(10, 1, "softmax")
])

nn.train(X, y)