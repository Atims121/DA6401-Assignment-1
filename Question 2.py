import numpy as np

class FeedforwardNeuralNetwork():
    def __init__(self, input_size, hidden_layers, output_size, neurons, acti_func, output_acti_func, weight_init, init_toggle):

        self.input_size, self.hidden_layers, self.output_size = input_size, hidden_layers, output_size
        self.neurons = neurons
        self.activation_function, self.output_activation_function = acti_func, output_acti_func
        self.weight_init = weight_init
        self.weights, self.biases =[], []

        if init_toggle:
            self.initialize_weights()
            self.initialize_biases()

    def initialize_weights(self):
        self.weights.append(np.random.randn(self.input_size, self.neurons))    
        
      #  self.layers = len(layer_sizes)
       # self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01 for i in range(len(layer_sizes) - 1)]
    
    def initialize_biases(self):

    def activation(self, x):

    def output_activation(self, x):




    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        #X: Input data (batch_size, 
        self.activations = [X]  # Storing activations for each layer
        for i in range(len(self.weights) - 1):
            X = self.sigmoid(np.dot(X, self.weights[i]) + self.biases[i])   #Using sigmoid function for all the hidden layers 
            self.activations.append(X)
        output = self.softmax(np.dot(X, self.weights[-1]) + self.biases[-1])       #Using softmax for the output layer
        self.activations.append(output)
        return output
        # Output probabilities (batch_size, num_classes)

