import numpy as np

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
        'function': lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum(),
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
    def __init__(self, layers: list, epochs: int = 5000, learning_rate: float = 0.001, early_stopping: bool = False):
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

    def cost(self, y_true, y_pred):
        '''Mean Squared Error cost function.'''
        return np.mean((y_true - y_pred) ** 2)
    
    def cost_derivative(self, y_true, y_pred):
        '''Derivative of the cost function.'''
        return y_pred - y_true
    
    def forward_propagation(self, X):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.output = layer.activation_function(X @ layer.weights + layer.biases)
            else:
                layer.output = layer.activation_function(self.layers[i - 1].output @ layer.weights + layer.biases)
        return self.layers[-1].output
    
    def back_propagation(self, X, y):
        '''Backpropagation algorithm to update weights and biases.'''
        deltas = list()
        for l in reversed(range(len(self.layers))):
            layer = self.layers[l]
            if (l == len(self.layers) - 1):
                deltas.insert(0, self.cost_derivative(y[0], layer.output) * layer.derivative_activation_function(layer.output))
            else:
                deltas.insert(0, (deltas[0] @ self.layers[l + 1].weights.T) * layer.derivative_activation_function(layer.output))

            layer.biases -= self.learning_rate * np.mean(deltas[0])
            layer.weights -= self.learning_rate * np.mean(deltas[0])

    def train(self, X, y):
        for epoch in range(self.epochs):
            y_pred = self.forward_propagation(X)
            self.back_propagation(X, y)
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {self.cost(y, y_pred)}")
