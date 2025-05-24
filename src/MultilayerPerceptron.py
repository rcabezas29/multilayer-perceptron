import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

activation_functions = {
    'sigmoid': {
        'function': lambda x: 1 / (1 + np.exp(-x)),
        'derivative': lambda x: x * (1 - x)
    },
    'relu': {
        'function': lambda x: np.maximum(0, x),
        'derivative': lambda x: np.where(x > 0, 1, 0)
    },
    'tanh': {
		'function': lambda x: np.tanh(x),
		'derivative': lambda x: 1 - np.tanh(x) ** 2
	},
    'leaky_relu': {
		'function': lambda x: np.where(x > 0, x, 0.01 * x),
		'derivative': lambda x: np.where(x > 0, 1, 0.01)
	},
    'elu': {
		'function': lambda x: np.where(x > 0, x, np.exp(x) - 1),
		'derivative': lambda x: np.where(x > 0, 1, np.exp(x))
	},
    'softmax': { # Softmax is not used as an activation function in hidden layers
		'function': lambda x: np.exp(x - np.max(x, axis=1, keepdims=True)) / np.sum(np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True),
		'derivative': lambda x: x
	}
}

class Layer:
    def __init__(self, n_inputs: int, n_neurons: int, activation_function: str):
        self.activation_function_name = activation_function
        self.activation_function = activation_functions[activation_function]['function']
        self.derivative_activation_function = activation_functions[activation_function]['derivative']
        self.weights = np.random.rand(n_inputs, n_neurons) * 2 - 1
        self.biases = np.random.rand(1, n_neurons) * 2 - 1
        self.output = np.zeros((1, n_neurons))

    def __repr__(self):
        return f"Layer(n_inputs={self.weights.shape[0]}, n_neurons={self.weights.shape[1]}, activation_function='{self.activation_function_name}')"
    
    def __str__(self):
        return f"Layer with {self.weights.shape[0]} inputs, {self.weights.shape[1]} neurons, using '{self.activation_function_name}' activation function."

class MultilayerPerceptron:
    def __init__(self, layers: list = [],
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
        :param verbose: Whether to print training progress.
        :param adam: Whether to use Adam optimization.
        """
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.adam = adam
        self.decay_rates = (0.9, 0.9)
        self.mean_momentum = [np.zeros(layer.weights.shape) for layer in layers]
        self.var_momentum = [np.zeros(layer.weights.shape) for layer in layers]

    def softmax_crossentropy_with_logits(self, reference_answers, logits):
        '''Softmax cross-entropy cost function.'''
        return np.mean(-np.sum(reference_answers * np.log(logits + 1e-15), axis=1))

    def grad_softmax_crossentropy_with_logits(self, reference_answers, logits):
        '''Gradient of the softmax cross-entropy cost function.'''
        return (activation_functions['softmax']['function'](logits) - reference_answers) / logits.shape[0]

    def feedforward(self, X):
        for l, layer in enumerate(self.layers):
            z = X @ layer.weights + layer.biases if l == 0 else self.layers[l - 1].output @ layer.weights + layer.biases
            layer.output = layer.activation_function(z)
        return self.layers[-1].output

    def backpropagation(self, X, y):
        '''Backpropagation algorithm to update weights and biases.'''
        deltas = [None] * len(self.layers)
        logits = self.layers[-2].output @ self.layers[-1].weights + self.layers[-1].biases
    
        for l in reversed(range(len(self.layers))):
            layer = self.layers[l]
            if (l == len(self.layers) - 1):
                deltas[-1] = self.grad_softmax_crossentropy_with_logits(y, logits)
            else:
                deltas[l] = (deltas[l + 1] @ self.layers[l + 1].weights.T) * layer.derivative_activation_function(layer.output)
            
            grad_w = (X if l == 0 else self.layers[l - 1].output).T @ deltas[l]
            if self.adam:
                self.mean_momentum[l] = self.decay_rates[0] * self.mean_momentum[l] + (1 - self.decay_rates[0]) * grad_w
                self.var_momentum[l] = self.decay_rates[1] * self.var_momentum[l] + (1 - self.decay_rates[1]) * (grad_w ** 2)
                mean_momentum_hat = self.mean_momentum[l] / (1 - self.decay_rates[0] ** (self.epochs + 1))
                var_momentum_hat = self.var_momentum[l] / (1 - self.decay_rates[1] ** (self.epochs + 1))
                layer.weights -= self.learning_rate * mean_momentum_hat / (np.sqrt(var_momentum_hat) + 1e-15)
            else:
                layer.weights -= self.learning_rate * grad_w
            layer.biases -= self.learning_rate * np.mean(deltas[l])

    def train(self, X, y):
        '''Train the MLP using backpropagation and gradient descent.'''
        loss = []
        accuracy = []
        for _ in tqdm(range(self.epochs)):
            y_pred = self.feedforward(X)
            self.backpropagation(X, y)
            loss.append(self.softmax_crossentropy_with_logits(y, y_pred))
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
            
    def train_with_validation(self, X_train, y_train, X_val, y_val, show_plots=True):
        '''Train the MLP using backpropagation and gradient descent with validation.'''
        loss = []
        accuracy = []
        val_loss = []
        val_accuracy = []
        for epoch in range(self.epochs):
            y_pred = self.feedforward(X_train)
            self.backpropagation(X_train, y_train)
            loss.append(self.softmax_crossentropy_with_logits(y_train, y_pred))
            val_loss.append(self.softmax_crossentropy_with_logits(y_val, self.predict(X_val)))

            if self.verbose:
                print(f"epoch {epoch+1:02d}/{self.epochs} - loss: {loss[-1]:.4f}  - val_loss: {val_loss[-1]:4f}", end='\r')

            if show_plots:
                accuracy.append(self.evaluate(X_train, y_train))
                val_accuracy.append(self.evaluate(X_val, y_val))
            if self.early_stopping and loss[-1] < 5e-3:
                break

        if show_plots:
            _, axs = plt.subplots(1, 2, figsize=(12, 5))
            axs[0].plot(loss, label='Training Loss', color='blue')
            axs[0].plot(val_loss, label='Validation Loss', color='orange')
            axs[0].set_title('Loss over epochs')
            axs[0].set_xlabel('Epochs')
            axs[0].set_ylabel('Loss')
            axs[0].legend()
            
            axs[1].plot(accuracy, label='Training Accuracy', color='blue')
            axs[1].plot(val_accuracy, label='Validation Accuracy', color='orange')
            axs[1].set_title('Accuracy over epochs')
            axs[1].set_xlabel('Epochs')
            axs[1].set_ylabel('Accuracy')
            axs[1].legend()
            
            plt.tight_layout()
            plt.show()

    def predict(self, X):
        output = X
        for i, layer in enumerate(self.layers):
            z = output @ layer.weights + layer.biases
            output = layer.activation_function(z)
        return output

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y, axis=1)
        return np.mean(y_pred_labels == y_true_labels)

    def save(self, filename='saved_model.json'):
        '''Save the model to a file.'''
        json_data = []
        for layer in self.layers:
            l = {
                'weights': layer.weights.tolist(),
                'biases': layer.biases.tolist(),
                'activation_function': layer.activation_function_name
            }
            json_data.append(l)
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"saving model './{filename}' to disk...")

    def load(self, filename):
        '''Load the model from a file.'''
        with open(filename, 'r') as f:
            json_data = json.load(f)
        self.layers = []
        for layer_data in json_data:
            layer = Layer(
                n_inputs=len(layer_data['weights']),
                n_neurons=len(layer_data['weights'][0]),
                activation_function=layer_data['activation_function']
            )
            layer.weights = np.array(layer_data['weights'])
            layer.biases = np.array(layer_data['biases'])
            self.layers.append(layer)
