import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

activation_functions = {
    'sigmoid': {
        'function': lambda x: 1 / (1 + np.exp(-x)),
        'derivative': lambda x: x * (1 - x)
    },
    'relu': {
        'function': lambda x: np.maximum(0, x),
        'derivative': lambda x: np.where(x > 0, 1, 0)
    },
    'softmax': {
        'function': lambda x: np.exp(x) / np.exp(x).sum(),
        'derivative': lambda x: x * (1 - x)
    }
}

class Layer:
    def __init__(self, n_inputs: int, n_neurons: int, activation_function: str):
        self.activation_function = activation_functions[activation_function]['function']
        self.derivative_activation_function = activation_functions[activation_function]['derivative']
        self.weights = np.random.rand(n_inputs, n_neurons) * 2 - 1
        self.biases = np.random.rand(1, n_neurons) * 2 - 1
        self.output = np.zeros((1, n_neurons))

class MultilayerPerceptron:
    def __init__(self, layers: list,
                 epochs: int = 50000,
                 learning_rate: float = 0.05,
                 early_stopping: bool = False,
                 verbose: bool = False,
                 adam: bool = False):
        """
        Multilayer Perceptron constructor.
        :param layers: List of Layer objects.
        :param epochs: Number of training epochs.
        :param learning_rate: Learning rate for weight updates.
        :param early_stopping: Whether to use early stopping.
        """
        if len(layers) < 2:
            raise ValueError("MultilayerPerceptron requires at least two Layers.")
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.adam = adam
        self.momentum = 0.9
        self.velocity = [np.zeros(layer.weights.shape) for layer in layers]

    def binary_cross_entropy(self, y_true, y_pred):
        '''Binary cross-entropy cost function.'''
        return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
    
    def binary_cross_entropy_derivative(self, y_true, y_pred):
        '''Derivative of the binary cross-entropy cost function.'''
        return -(y_true / y_pred + 1e-15) + (1 - y_true) / (1 - y_pred + 1e-15)

    def feedforward(self, X):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.output = layer.activation_function(X @ layer.weights + layer.biases)
            else:
                layer.output = layer.activation_function(self.layers[i - 1].output @ layer.weights + layer.biases)
        return self.layers[-1].output
    
    def backpropagation(self, X, y):
        '''Backpropagation algorithm to update weights and biases.'''
        deltas = list()
        for l in reversed(range(len(self.layers))):
            layer = self.layers[l]
            if (l == len(self.layers) - 1):
                deltas.insert(0, self.binary_cross_entropy_derivative(y, layer.output) * layer.derivative_activation_function(layer.output))
            else:
                deltas.insert(0, (deltas[0] @ self.layers[l + 1].weights.T) * layer.derivative_activation_function(layer.output))

            if self.adam:
                self.velocity[l] = self.momentum * self.velocity[l] + (1 - self.momentum) * (X if l == 0 else self.layers[l - 1].output).T @ deltas[0]
                layer.weights -= self.learning_rate * self.velocity[l]
                layer.biases -= self.learning_rate * np.mean(deltas[0])
            else:
                layer.weights -= self.learning_rate * (X if l == 0 else self.layers[l - 1].output).T @ deltas[0]
                layer.biases -= self.learning_rate * np.mean(deltas[0], axis=0, keepdims=True)

    def train(self, X, y):
        loss = []
        accuracy = []
        for _ in tqdm(range(self.epochs)):
            y_pred = self.feedforward(X)
            self.backpropagation(X, y)
            loss.append(self.binary_cross_entropy(y, y_pred))
            if self.verbose:
                accuracy.append(self.evaluate(X, y))
            if self.early_stopping and loss[-1] < 5e-3:
                break

        if self.verbose:
            _, axs = plt.subplots(1, 2, figsize=(12, 5))
            axs[0].plot(loss)
            axs[0].set_title('Loss over epochs')
            axs[0].set_xlabel('Epochs')
            axs[0].set_ylabel('Loss')
            axs[1].plot(accuracy)
            axs[1].set_title('Accuracy over epochs')
            axs[1].set_xlabel('Epochs')
            axs[1].set_ylabel('Accuracy')
            plt.tight_layout()
            plt.show()
            

    def predict(self, X):
        """
        Predict the output for the given input data.
        :param X: The input data.
        :return: The predicted output.
        """
        output = list()
        for i, layer in enumerate(self.layers):
            if i == 0:
                output.append(layer.activation_function(X @ layer.weights + layer.biases))
            else:
                output.append(layer.activation_function(output[-1] @ layer.weights + layer.biases))
        return output[-1].flatten()

    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.
        :param X: The input data.
        :param y: The true labels.
        :return: The accuracy of the model.
        """
        y_pred = np.round(self.predict(X)).astype(int)
        y_true = np.round(y).astype(int)

        return np.mean(y_pred == y_true.flatten())
