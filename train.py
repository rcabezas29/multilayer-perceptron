from MultilayerPerceptron import MultilayerPerceptron, Layer
import numpy as np
import pandas as pd

def standarize(X) -> np.ndarray:
    """
    Standardize the dataset by removing the mean and scaling to unit variance.
    :param X: The input data.
    :return: The standardized data.
    """
    return (X - X.mean()) / X.std()

def	normalize(X) -> np.ndarray:
	"""
	Normalize the dataset to a range of [0, 1].
	:param X: The input data.
	:return: The normalized data.
	"""
	return (X - X.min()) / (X.max() - X.min())

df = pd.read_csv("train.csv", header=None)
y = df.iloc[:, 1].apply(lambda x: 1 if x == "M" else 0)
X = df.iloc[:, 2:]
X = normalize(X)

nn = MultilayerPerceptron([
    Layer(X.shape[1], 1, "sigmoid"),
    Layer(1, 1, "sigmoid")
], epochs=2000, learning_rate=0.2, early_stopping=False, verbose=True)

nn.train(X, y)

df = pd.read_csv("test.csv", header=None)
y_test = df.iloc[:, 1].apply(lambda x: 1 if x == "M" else 0)
X_test = df.iloc[:, 2:]
X_test = normalize(X_test)

accuracy = nn.evaluate(X_test.values, y_test.values)
print(f"Accuracy: {accuracy * 100:.2f}%")
