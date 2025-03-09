import numpy as np

class FeedforwardNeuralNetwork:
    def __init__(self, layer_sizes):
        #layer_sizes: List defining number of neurons in each layer, helps in changing the number of hidden layers and number of neurons in each hidden layer.
        self.layers = len(layer_sizes)
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01 for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros((1, layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        #X: Input data (batch_size, input_features)
        self.activations = [X]  # Storing activations for each layer
        for i in range(len(self.weights) - 1):
            X = self.sigmoid(np.dot(X, self.weights[i]) + self.biases[i])   #Using sigmoid function for all the hidden layers 
            self.activations.append(X)
        output = self.softmax(np.dot(X, self.weights[-1]) + self.biases[-1])       #Using softmax for the output layer
        self.activations.append(output)
        return output
        # Output probabilities (batch_size, num_classes)

