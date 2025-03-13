import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exps = np.exp(x - x.max())
    return exps / exps.sum()

class   Layer:
    def __init__(self, n_inputs: int, n_neurons: int, activation_function: str):
        self.activation_function = activation_function
        self.weights = np.random.rand(n_inputs, n_neurons) * 2 - 1
        self.biases = np.random.rand(1, n_neurons) * 2 - 1
        self.output = np.zeros((1, n_neurons))

class   MultilayerPerceptron:
    def __init__(self, layers: list, epochs: int = 5000, learning_rate: float = 0.1, early_stopping: bool = False):
        if len(layers) < 2:
            raise ValueError("MultilayerPerceptron requires at least two Layers.")
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate

    def cost(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def forward_propagation(self, X):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.output = layer.activation_function(X @ layer.weights + layer.biases)
            else:
                layer.output = layer.activation_function(self.layers[i - 1].output @ layer.weights + layer.biases)
        return self.layers[-1].output
    
    def back_propagation(self, X, y):
        '''Backpropagation algorithm to update weights and biases.'''
        deltas = []
        for i, layer in reversed(list(enumerate(self.layers))):
            if i == len(self.layers) - 1:
                error = y - layer.output
                deltas.insert(0, error * layer.output * (1 - layer.output))
            else:
                deltas.insert(0, deltas[0] @ self.layers[i + 1].weights.T * layer.output * (1 - layer.output))
        

    def train(self, X, y):
        for epoch in range(self.epochs):
            y_pred = self.forward_propagation(X)
            self.back_propagation(X, y)
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {self.cost(y, y_pred)}")
