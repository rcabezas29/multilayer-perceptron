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
    def __init__(self, layers: list, epochs: int = 50000, learning_rate: float = 0.05, early_stopping: bool = False, verbose: bool = False):
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

    def cost(self, y_true, y_pred):
        '''Mean Squared Error cost function.'''
        return np.mean((y_true - y_pred) ** 2)
    
    def cost_derivative(self, y_true, y_pred):
        '''Derivative of the cost function.'''
        return y_pred - y_true
    
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
                deltas.insert(0, self.cost_derivative(y[0], layer.output) * layer.derivative_activation_function(layer.output))
            else:
                deltas.insert(0, (deltas[0] @ self.layers[l + 1].weights.T) * layer.derivative_activation_function(layer.output))

            layer.weights -= self.learning_rate * (X if l == 0 else self.layers[l - 1].output).T @ deltas[0]
            layer.biases -= self.learning_rate * np.mean(deltas[0])

    def train(self, X, y):
        loss = []
        for epoch in range(self.epochs):
            y_pred = self.feedforward(X)
            self.backpropagation(X, y)
            if epoch % 100 == 0 and self.verbose:
                loss.append(self.cost(y, y_pred))
                print(f"Epoch: {epoch}, Loss: {loss[-1]}")

    def predict(self, X):
        """
        Predict the output for the given input data.
        :param X: The input data.
        :return: The predicted output.
        """
        for layer in self.layers:
            X = layer.activation_function(X @ layer.weights + layer.biases)
        return X

    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.
        :param X: The input data.
        :param y: The true labels.
        :return: The accuracy of the model.
        """
        y_pred = self.predict(X)
        y_pred = np.round(y_pred)
        y_true = y.reshape(y_pred.shape)

        accuracy = np.mean(y_pred == y_true)
        return accuracy
