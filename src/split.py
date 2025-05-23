import pandas as pd
import numpy as np

def split_dataset(df: pd.DataFrame) -> tuple:
	"""Split the dataset into training and testing sets."""
	msk = np.random.rand(len(df)) <= 0.8
	train = df[msk]
	test = df[~msk]
	return train.copy(), test.copy()
