import numpy as np

class   Loss:
    def loss(self, y_true, y_pred):
        raise NotImplementedError("This method should be overridden by subclasses")
    def gradient(self, y_true, y_pred):
        raise NotImplementedError("This method should be overridden by subclasses")

class MeanSquaredError(Loss):
    def loss(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()

    def gradient(self, y_true, y_pred):
        return -2 * (y_true - y_pred) / len(y_true)

class BinaryCrossentropy(Loss):
    def loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def gradient(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred)) / len(y_true)

class CategoricalCrossentropy(Loss):
    def loss(self, y_true, y_pred):
        return np.mean(-np.sum(y_true * np.log(y_pred + 1e-15), axis=1))

    def gradient(self, y_true, y_pred):
        return (y_pred - y_true) / y_pred.shape[0]
