from src.MultilayerPerceptron import MultilayerPerceptron, Layer
from src.utils import standardize, to_categorical
from src.split import split_dataset
import pandas as pd
import sys
import os

if os.path.exists("train.csv") and os.path.exists("test.csv"):
    df_train = pd.read_csv("train.csv").copy()
    df_test = pd.read_csv("test.csv").copy()
else:
    # Read the dataset from a CSV file as parameter
	if len(sys.argv) != 2:
		print("Usage: python train.py <path_to_csv_file>")
		sys.exit(1)
	file_path = sys.argv[1]
	df = pd.read_csv(file_path)
	df_train, df_test = split_dataset(df)

df_train.columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
df_train['diagnosis'] = df_train['diagnosis'].map({'M': 1, 'B': 0})
df_train.drop(columns=['id'], inplace=True)

X_train = df_train.drop(columns='diagnosis').to_numpy()
y_train = df_train['diagnosis'].to_numpy().reshape(-1, 1)
y_train = to_categorical(y_train, num_classes=2)

X_train = standardize(X_train)

nn = MultilayerPerceptron([
    Layer(X_train.shape[1], 25, "sigmoid"),
    Layer(25, 15, "relu"),
    Layer(15, 2, "softmax")
], epochs=3000, learning_rate=0.005, early_stopping=True, verbose=True, adam=True)

df_test.columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
df_test['diagnosis'] = df_test['diagnosis'].map({'M': 1, 'B': 0})
df_test.drop(columns=['id'], inplace=True)

X_test = df_test.drop(columns='diagnosis').to_numpy()
y_test = df_test['diagnosis'].to_numpy().reshape(-1, 1)
y_test = to_categorical(y_test, num_classes=2)

X_test = standardize(X_test)

nn.train_with_validation(X_train, y_train, X_test, y_test)

nn.save()