from MultilayerPerceptron import MultilayerPerceptron, Layer
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def to_categorical(y, num_classes=2):
	"""Convert class vector (integers) to binary class matrix."""
	return np.eye(num_classes)[y.reshape(-1)]

df_train = pd.read_csv('./train.csv', header=None)
df_train.columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
df_train['diagnosis'] = df_train['diagnosis'].map({'M': 1, 'B': 0})
df_train.drop(columns=['id'], inplace=True)

X_train = df_train.drop(columns='diagnosis').to_numpy()
y_train = df_train['diagnosis'].to_numpy().reshape(-1, 1)
y_train = to_categorical(y_train, num_classes=2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

nn = MultilayerPerceptron([
    Layer(X_train.shape[1], 25, "sigmoid"),
    Layer(25, 15, "relu"),
    Layer(15, 2, "softmax")
], epochs=1000, learning_rate=0.001, early_stopping=False, verbose=True, adam=True)

nn.train(X_train, y_train)

#####################################################

df_test = pd.read_csv('./test.csv', header=None)
df_test.columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
df_test['diagnosis'] = df_test['diagnosis'].map({'M': 1, 'B': 0})
df_test.drop(columns=['id'], inplace=True)

X_test = df_test.drop(columns='diagnosis').to_numpy()
y_test = df_test['diagnosis'].to_numpy().reshape(-1, 1)
y_test = to_categorical(y_test, num_classes=2)

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

accuracy = nn.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
