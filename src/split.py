import pandas as pd
import numpy as np
import sys	

def split_dataset(df: pd.DataFrame) -> tuple:
	"""Split the dataset into training and testing sets."""
	msk = np.random.rand(len(df)) <= 0.8
	train = df[msk]
	test = df[~msk]
	return train.copy(), test.copy()

if __name__ == "__main__":
	# Read the dataset from a CSV file as parameter
	if len(sys.argv) != 2:
		print("Usage: python split.py <path_to_csv_file>")
		sys.exit(1)
	file_path = sys.argv[1]
	# Load the dataset
	df = pd.read_csv(file_path)
	# Split the dataset
	train, test = split_dataset(df)
	# Save the training and testing sets to CSV files
	train.to_csv("train.csv", index=False)
	test.to_csv("test.csv", index=False)
