import pandas as pd
import numpy as np
import sys

if len(sys.argv) != 2:
	print("Usage: python split.py dataset.csv")
	sys.exit(1)

if not sys.argv[1].endswith(".csv"):
	print("The dataset must be in CSV format.")
	sys.exit(1)

df = pd.read_csv(sys.argv[1])
msk = np.random.rand(len(df)) <= 0.8
train = df[msk]
test = df[~msk]
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)
