from src.MultilayerPerceptron import MultilayerPerceptron, Layer
from src.utils import standardize, to_categorical
import pandas as pd
import os

if os.path.exists("test.csv") and os.path.exists("saved_model.json"):
    df_test = pd.read_csv("test.csv").copy()
else:
    print("Please run the training script first to generate 'test.csv' and 'saved_model.json'.")
    exit(1)

nn = MultilayerPerceptron()
nn.load('saved_model.json')

df_test.columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
df_test['diagnosis'] = df_test['diagnosis'].map({'M': 1, 'B': 0})
df_test.drop(columns=['id'], inplace=True)

X_test = df_test.drop(columns='diagnosis').to_numpy()
y_test = df_test['diagnosis'].to_numpy().reshape(-1, 1)
y_test = to_categorical(y_test, num_classes=2)

X_test = standardize(X_test)

accuracy = nn.evaluate(X_test, y_test)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
