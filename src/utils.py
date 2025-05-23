import numpy as np

def to_categorical(y, num_classes=2):
	"""Convert class vector (integers) to binary class matrix."""
	return np.eye(num_classes)[y.reshape(-1)]

def	standardize(X):
	"""Standardize features by removing the mean and scaling to unit variance."""
	mean = np.mean(X, axis=0)
	std = np.std(X, axis=0)
	X_std = (X - mean) / std
	return X_std
