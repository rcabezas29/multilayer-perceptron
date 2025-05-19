from MultilayerPerceptron import MultilayerPerceptron, Layer
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

df_train = pd.read_csv('./train.csv', header=None)
df_train.columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
df_train['diagnosis'] = df_train['diagnosis'].map({'M': 1, 'B': 0})
df_train.drop(columns=['id'], inplace=True)

X_train = df_train.drop(columns='diagnosis').to_numpy()
y_train = df_train['diagnosis'].to_numpy().reshape(-1, 1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

nn = MultilayerPerceptron([
    Layer(X_train.shape[1], 25, "sigmoid"),
    Layer(25, 15, "relu"),
    Layer(15, 1, "sigmoid")
], epochs=100, learning_rate=0.001, early_stopping=False, verbose=True)

nn.train(X_train, y_train)

#####################################################

df_test = pd.read_csv('./test.csv', header=None)
df_test.columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
df_test['diagnosis'] = df_test['diagnosis'].map({'M': 1, 'B': 0})
df_test.drop(columns=['id'], inplace=True)

X_test = df_test.drop(columns='diagnosis').to_numpy()
y_test = df_test['diagnosis'].to_numpy().reshape(-1, 1)

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

accuracy = nn.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
