class Backpropagation():
    def __init__(self, 
                 nn: FeedforwardNeuralNetwork, 
                 loss="cross_entropy", 
                 act_func="sigmoid"):
        
        self.nn, self.loss, self.activation_function = nn, loss, act_func
    
    def loss_derivative(self, y, y_pred):
        if self.loss == "cross_entropy":
            return -y / y_pred
        elif self.loss == "mean_squared_error":
            return (y_pred - y)
        else:
            raise Exception("Invalid loss function")
        
    def activation_derivative(self, x):
        # x is the post-activation value
        if self.activation_function == "sigmoid":
            return x * (1 - x)
        elif self.activation_function == "tanh":
            return 1 - x ** 2
        elif self.activation_function == "relu":
            return (x > 0).astype(int)
        elif self.activation_function == "identity":
            return np.ones(x.shape)
        else:
            raise Exception("Invalid activation function")
        
    def output_activation_derivative(self, y, y_pred):
        if self.nn.output_activation_function == "softmax":
            # derivative of softmax is a matrix
            return np.diag(y_pred) - np.outer(y_pred, y_pred)
        else:
            raise Exception("Invalid output activation function")

    def backward(self, y, y_pred):
        self.d_weights, self.d_biases = [], []
        self.d_h, self.d_a = [], []

        self.d_h.append(self.loss_derivative(y, y_pred))
        output_derivative_matrix = []
        for i in range(y_pred.shape[0]):
            output_derivative_matrix.append(np.matmul(self.loss_derivative(y[i], y_pred[i]), self.output_activation_derivative(y[i], y_pred[i])))
        self.d_a.append(np.array(output_derivative_matrix))

        for i in range(self.nn.hidden_layers, 0, -1):
            self.d_weights.append(np.matmul(self.nn.post_activation[i].T, self.d_a[-1]))
            self.d_biases.append(np.sum(self.d_a[-1], axis=0))
            self.d_h.append(np.matmul(self.d_a[-1], self.nn.weights[i].T))
            self.d_a.append(self.d_h[-1] * self.activation_derivative(self.nn.post_activation[i]))

        self.d_weights.append(np.matmul(self.nn.post_activation[0].T, self.d_a[-1]))
        self.d_biases.append(np.sum(self.d_a[-1], axis=0))

        self.d_weights.reverse()
        self.d_biases.reverse()
        for i in range(len(self.d_weights)):
            self.d_weights[i] = self.d_weights[i] / y.shape[0]
            self.d_biases[i] = self.d_biases[i] / y.shape[0]

        return self.d_weights, self.d_biases
    
# Optimizers
class Optimizer():
    def __init__(self, 
                 nn: FeedforwardNeuralNetwork, 
                 bp:Backpropagation, 
                 lr=0.001, 
                 optimizer="sgd", 
                 momentum=0.9,
                 epsilon=1e-8,
                 beta=0.9,
                 beta1=0.9,
                 beta2=0.999, 
                 t=0,
                 decay=0):
        
        self.nn, self.bp, self.lr, self.optimizer = nn, bp, lr, optimizer
        self.momentum, self.epsilon, self.beta1, self.beta2, self.beta = momentum, epsilon, beta1, beta2, beta
        self.h_weights = [np.zeros_like(w) for w in self.nn.weights]
        self.h_biases = [np.zeros_like(b) for b in self.nn.biases]
        self.hm_weights = [np.zeros_like(w) for w in self.nn.weights]
        self.hm_biases = [np.zeros_like(b) for b in self.nn.biases]
        self.t = t
        self.decay = decay

    def run(self, d_weights, d_biases):
        if(self.optimizer == "sgd"):
            self.SGD(d_weights, d_biases)
        elif(self.optimizer == "momentum"):
            self.MomentumGD(d_weights, d_biases)
        elif(self.optimizer == "nag"):
            self.NAG(d_weights, d_biases)
        elif(self.optimizer == "rmsprop"):
            self.RMSProp(d_weights, d_biases)
        elif(self.optimizer == "adam"):
            self.Adam(d_weights, d_biases)
        elif (self.optimizer == "nadam"):
            self.NAdam(d_weights, d_biases)
        else:
            raise Exception("Invalid optimizer")
    
    def SGD(self, d_weights, d_biases):
        for i in range(self.nn.hidden_layers + 1):
            self.nn.weights[i] -= self.lr * (d_weights[i] + self.decay * self.nn.weights[i])
            self.nn.biases[i] -= self.lr * (d_biases[i] + self.decay * self.nn.biases[i])

    def MomentumGD(self, d_weights, d_biases):
        for i in range(self.nn.hidden_layers + 1):
            self.h_weights[i] = self.momentum * self.h_weights[i] + d_weights[i]
            self.h_biases[i] = self.momentum * self.h_biases[i] + d_biases[i]

            self.nn.weights[i] -= self.lr * (self.h_weights[i] + self.decay * self.nn.weights[i])
            self.nn.biases[i] -= self.lr * (self.h_biases[i] + self.decay * self.nn.biases[i])

    def NesterovAG(self, d_weights, d_biases):        
        for i in range(self.nn.hidden_layers + 1):
            self.h_weights[i] = self.momentum * self.h_weights[i] + d_weights[i]
            self.h_biases[i] = self.momentum * self.h_biases[i] + d_biases[i]

            self.nn.weights[i] -= self.lr * (self.momentum * self.h_weights[i] + d_weights[i] + self.decay * self.nn.weights[i])
            self.nn.biases[i] -= self.lr * (self.momentum * self.h_biases[i] + d_biases[i] + self.decay * self.nn.biases[i])

    def RMSProp(self, d_weights, d_biases):
        for i in range(self.nn.hidden_layers + 1):
            self.h_weights[i] = self.beta * self.h_weights[i] + (1 - self.beta) * d_weights[i]**2
            self.h_biases[i] = self.beta * self.h_biases[i] + (1 - self.beta) * d_biases[i]**2

            self.nn.weights[i] -= (self.lr / (np.sqrt(self.h_weights[i]) + self.epsilon)) * d_weights[i] + self.decay * self.nn.weights[i] * self.lr
            self.nn.biases[i] -= (self.lr / (np.sqrt(self.h_biases[i]) + self.epsilon)) * d_biases[i] + self.decay * self.nn.biases[i] * self.lr

    def Adam(self, d_weights, d_biases):
        for i in range(self.nn.hidden_layers + 1):
            self.hm_weights[i] = self.beta1 * self.hm_weights[i] + (1 - self.beta1) * d_weights[i]
            self.hm_biases[i] = self.beta1 * self.hm_biases[i] + (1 - self.beta1) * d_biases[i]

            self.h_weights[i] = self.beta2 * self.h_weights[i] + (1 - self.beta2) * d_weights[i]**2
            self.h_biases[i] = self.beta2 * self.h_biases[i] + (1 - self.beta2) * d_biases[i]**2

            self.hm_weights_hat = self.hm_weights[i] / (1 - self.beta1**(self.t + 1))
            self.hm_biases_hat = self.hm_biases[i] / (1 - self.beta1**(self.t + 1))

            self.h_weights_hat = self.h_weights[i] / (1 - self.beta2**(self.t + 1))
            self.h_biases_hat = self.h_biases[i] / (1 - self.beta2**(self.t + 1))

            self.nn.weights[i] -= self.lr * (self.hm_weights_hat / ((np.sqrt(self.h_weights_hat)) + self.epsilon)) + self.decay * self.nn.weights[i] * self.lr
            self.nn.biases[i] -= self.lr * (self.hm_biases_hat / ((np.sqrt(self.h_biases_hat)) + self.epsilon)) + self.decay * self.nn.biases[i] * self.lr

    def NAdam(self, d_weights, d_biases):
        for i in range(self.nn.hidden_layers + 1):
            self.hm_weights[i] = self.beta1 * self.hm_weights[i] + (1 - self.beta1) * d_weights[i]
            self.hm_biases[i] = self.beta1 * self.hm_biases[i] + (1 - self.beta1) * d_biases[i]

            self.h_weights[i] = self.beta2 * self.h_weights[i] + (1 - self.beta2) * d_weights[i]**2
            self.h_biases[i] = self.beta2 * self.h_biases[i] + (1 - self.beta2) * d_biases[i]**2

            self.hm_weights_hat = self.hm_weights[i] / (1 - self.beta1 ** (self.t + 1))
            self.hm_biases_hat = self.hm_biases[i] / (1 - self.beta1 ** (self.t + 1))

            self.h_weights_hat = self.h_weights[i] / (1 - self.beta2 ** (self.t + 1))
            self.h_biases_hat = self.h_biases[i] / (1 - self.beta2 ** (self.t + 1))

            temp_update_w = self.beta1 * self.hm_weights_hat + ((1 - self.beta1) / (1 - self.beta1 ** (self.t + 1))) * d_weights[i]
            temp_update_b = self.beta1 * self.hm_biases_hat + ((1 - self.beta1) / (1 - self.beta1 ** (self.t + 1))) * d_biases[i]

            self.nn.weights[i] -= self.lr * (temp_update_w / ((np.sqrt(self.h_weights_hat)) + self.epsilon)) + self.decay * self.nn.weights[i] * self.lr
            self.nn.biases[i] -= self.lr * (temp_update_b / ((np.sqrt(self.h_biases_hat)) + self.epsilon)) + self.decay * self.nn.biases[i] * self.lr
