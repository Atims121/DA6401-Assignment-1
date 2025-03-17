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

    def initialize_biases(self):
        for _ in range(self.hidden_layers):
            self.biases.append(np.zeros(self.neurons))
        self.biases.append(np.zeros(self.output_size))
           
    def initialize_weights(self):
        self.weights.append(np.random.randn(self.input_size, self.neurons))
        for _ in range(self.hidden_layers - 1):
            self.weights.append(np.random.randn(self.neurons, self.neurons))
        self.weights.append(np.random.randn(self.neurons, self.output_size))

        if self.weight_init == "xavier":
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] * np.sqrt(1 / self.weights[i].shape[0])
        
        if(self.weight_init != "random" and self.weight_init != "xavier"):
            raise Exception("Invalid weight initialization method")
     
    def activation(self, x):
        if self.activation_function == "tanh":
            return np.tanh(x)
        elif self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == "relu":
            return np.maximum(0, x)
        else:
            raise Exception("Invalid activation function")

    def output_activation(self, x):
        if self.output_activation_function == "softmax":
            max_x = np.max(x, axis=1)
            max_x = max_x.reshape(max_x.shape[0], 1)
            exp_x = np.exp(x - max_x)
            softmax_mat = exp_x / np.sum(exp_x, axis=1).reshape(exp_x.shape[0], 1)
            return softmax_mat
        else:
            raise Exception("Invalid output activation function")
    
    def forward(self, X):
        self.pre_activation, self.post_activation = [x], [x]

        for i in range(self.hidden_layers):
            self.pre_activation.append(np.matmul(self.post_activation[-1], self.weights[i]) + self.biases[i])
            self.post_activation.append(self.activation(self.pre_activation[-1]))
            
        self.pre_activation.append(np.matmul(self.post_activation[-1], self.weights[-1]) + self.biases[-1])
        self.post_activation.append(self.output_activation(self.pre_activation[-1]))
        return self.post_activation[-1]


