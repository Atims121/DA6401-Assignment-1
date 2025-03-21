import wandb
import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
import wandb
import argparse

# Class names for the Fashion MNIST dataset
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Settings for the run
WANDB_PROJECT = "myprojectname"
WANDB_ENTITY = "myname"

parameters_dict = {
    "wandb_project": WANDB_PROJECT,
    "wandb_entity": WANDB_ENTITY,
    "dataset": DATASET,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "loss": LOSS,
    "optimizer": OPTIMIZER,
    "learning_rate": LEARNING_RATE,
    "momentum": MOMENTUM,
    "beta": BETA,
    "beta1": BETA1,
    "beta2": BETA2,
    "epsilon": EPSILON,
    "weight_decay": WEIGHT_DECAY,
    "weight_init": WEIGHT_INIT,
    "num_layers": NUM_LAYERS,
    "hidden_size": HIDDEN_SIZE,
    "activation": ACTIVATION
}

# Parse arguments and update parameters_dict
parser = argparse.ArgumentParser()
parser.add_argument("-wp", "--wandb_project", type=str, default=WANDB_PROJECT, help="wandb_project", required=True)
parser.add_argument("-we", "--wandb_entity", type=str, default=WANDB_ENTITY, help="wandb_entity", required=True)
parser.add_argument("-d", "--dataset", type=str, default=DATASET, help="Dataset to use choices=['fashion_mnist', 'mnist']")
parser.add_argument("-e", "--epochs", type=int, default=EPOCHS, help="Number of epochs")
parser.add_argument("-b", "--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
parser.add_argument("-l", "--loss", type=str, default=LOSS, help="Loss function to use choices=['cross_entropy', 'mean_squared_error']")
parser.add_argument("-o", "--optimizer", type=str, default=OPTIMIZER, help="Optimizer to use choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']")
parser.add_argument("-lr", "--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate")
parser.add_argument("-m", "--momentum", type=float, default=MOMENTUM, help="Momentum for Momentum and NAG")
parser.add_argument("-beta", "--beta", type=float, default=BETA, help="Beta for RMSProp")
parser.add_argument("-beta1", "--beta1", type=float, default=BETA1, help="Beta1 for Adam and Nadam")
parser.add_argument("-beta2", "--beta2", type=float, default=BETA2, help="Beta2 for Adam and Nadam")
parser.add_argument("-eps", "--epsilon", type=float, default=EPSILON, help="Epsilon for Adam and Nadam")
parser.add_argument("-w_d", "--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay")
parser.add_argument("-w_i", "--weight_init", type=str, default=WEIGHT_INIT, help="Weight initialization choices=['random', 'xavier']")
parser.add_argument("-nhl", "--num_layers", type=int, default=NUM_LAYERS, help="Number of hidden layers")
parser.add_argument("-sz", "--hidden_size", type=int, default=HIDDEN_SIZE, help="Hidden size")
parser.add_argument("-a", "--activation", type=str, default=ACTIVATION, help="Activation function choices=['sigmoid', 'tanh', 'relu']")

args = parser.parse_args()
parameters_dict.update(vars(args))

# Print the parameters
print("Parameters:")
for key, value in parameters_dict.items():
    print(f"{key}: {value}")

# Feedforward neural network
class FeedforwardNeuralNetwork():
    def __init__(self, neurons=128,hidden_layers=4,input_size=784,output_size=10,acti_func="sigmoid",weight_init="random",output_acti_func="softmax",init_toggle=True):      
        self.neurons, self.hidden_layers = neurons, hidden_layers
        self.weights, self.biases = [], []
        self.input_size, self.output_size = input_size, output_size
        self.activation_function, self.weight_init = acti_func, weight_init
        self.output_activation_function = output_acti_func

        if init_toggle:
            self.initialize_weights()
            self.initiate_biases()
    
    def initiate_biases(self):
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
        if self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == "tanh":
            return np.tanh(x)
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
    
    def forward(self, x):
        self.pre_activation, self.post_activation = [x], [x]

        for i in range(self.hidden_layers):
            self.pre_activation.append(np.matmul(self.post_activation[-1], self.weights[i]) + self.biases[i])
            self.post_activation.append(self.activation(self.pre_activation[-1]))
            
        self.pre_activation.append(np.matmul(self.post_activation[-1], self.weights[-1]) + self.biases[-1])
        self.post_activation.append(self.output_activation(self.pre_activation[-1]))

        return self.post_activation[-1]
    
# Loss functions
def loss(loss, y, y_pred):
    if loss == "cross_entropy": # Cross Entropy
        return -np.sum(y * np.log(y_pred))
    elif loss == "mean_squared_error": # Mean Squared Error
        return np.sum((y - y_pred) ** 2) / 2
    else:
        raise Exception("Invalid loss function")
    
# Backpropagation
class Backpropagation():
    def __init__(self,fnn:FeedforwardNeuralNetwork,loss="cross_entropy",acti_func="sigmoid"):
        self.fnn, self.loss, self.activation_function = fnn, loss, acti_func
    
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
        else:
            raise Exception("Invalid activation function")
        
    def output_activation_derivative(self, y, y_pred):
        if self.fnn.output_activation_function == "softmax":
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

        for i in range(self.fnn.hidden_layers, 0, -1):
            self.d_weights.append(np.matmul(self.fnn.post_activation[i].T, self.d_a[-1]))
            self.d_biases.append(np.sum(self.d_a[-1], axis=0))
            self.d_h.append(np.matmul(self.d_a[-1], self.fnn.weights[i].T))
            self.d_a.append(self.d_h[-1] * self.activation_derivative(self.fnn.post_activation[i]))

        self.d_weights.append(np.matmul(self.fnn.post_activation[0].T, self.d_a[-1]))
        self.d_biases.append(np.sum(self.d_a[-1], axis=0))

        self.d_weights.reverse()
        self.d_biases.reverse()
        for i in range(len(self.d_weights)):
            self.d_weights[i] = self.d_weights[i] / y.shape[0]
            self.d_biases[i] = self.d_biases[i] / y.shape[0]

        return self.d_weights, self.d_biases
    
# Optimizers
class Optimizer():
    def __init__(self,fnn: FeedforwardNeuralNetwork,bp:Backpropagation,lr=0.001,optimizer="sgd",momentum=0.9,epsilon=1e-8,beta=0.9,beta1=0.9,beta2=0.999,t=0,decay=0):
        self.fnn, self.bp, self.lr, self.optimizer = fnn, bp, lr, optimizer
        self.momentum, self.epsilon, self.beta1, self.beta2, self.beta = momentum, epsilon, beta1, beta2, beta
        self.h_weights = [np.zeros_like(w) for w in self.fnn.weights]
        self.h_biases = [np.zeros_like(b) for b in self.fnn.biases]
        self.hm_weights = [np.zeros_like(w) for w in self.fnn.weights]
        self.hm_biases = [np.zeros_like(b) for b in self.fnn.biases]
        self.t = t
        self.decay = decay

    def run(self, d_weights, d_biases):
        if(self.optimizer == "sgd"):
            self.SGD(d_weights, d_biases)
        elif(self.optimizer == "momentum"):
            self.MomentumGD(d_weights, d_biases)
        elif(self.optimizer == "nesterov"):
            self.NesterovAG(d_weights, d_biases)
        elif(self.optimizer == "rmsprop"):
            self.RMSProp(d_weights, d_biases)
        elif(self.optimizer == "adam"):
            self.Adam(d_weights, d_biases)
        elif (self.optimizer == "nadam"):
            self.NAdam(d_weights, d_biases)
        else:
            raise Exception("Invalid optimizer")
    
    def SGD(self, d_weights, d_biases):
        for i in range(self.fnn.hidden_layers + 1):
            self.fnnnn.weights[i] -= self.lr * (d_weights[i] + self.decay * self.fnn.weights[i])
            self.fnn.biases[i] -= self.lr * (d_biases[i] + self.decay * self.fnn.biases[i])

    def MomentumGD(self, d_weights, d_biases):
        for i in range(self.fnn.hidden_layers + 1):
            self.h_weights[i] = self.momentum * self.h_weights[i] + d_weights[i]
            self.h_biases[i] = self.momentum * self.h_biases[i] + d_biases[i]

            self.fnn.weights[i] -= self.lr * (self.h_weights[i] + self.decay * self.fnn.weights[i])
            self.fnn.biases[i] -= self.lr * (self.h_biases[i] + self.decay * self.fnn.biases[i])

    def NesterovAG(self, d_weights, d_biases):        
        for i in range(self.fnn.hidden_layers + 1):
            self.h_weights[i] = self.momentum * self.h_weights[i] + d_weights[i]
            self.h_biases[i] = self.momentum * self.h_biases[i] + d_biases[i]

            self.fnn.weights[i] -= self.lr * (self.momentum * self.h_weights[i] + d_weights[i] + self.decay * self.fnn.weights[i])
            self.fnn.biases[i] -= self.lr * (self.momentum * self.h_biases[i] + d_biases[i] + self.decay * self.fnn.biases[i])

    def RMSProp(self, d_weights, d_biases):
        for i in range(self.fnn.hidden_layers + 1):
            self.h_weights[i] = self.beta * self.h_weights[i] + (1 - self.beta) * d_weights[i]**2
            self.h_biases[i] = self.beta * self.h_biases[i] + (1 - self.beta) * d_biases[i]**2

            self.fnn.weights[i] -= (self.lr / (np.sqrt(self.h_weights[i]) + self.epsilon)) * d_weights[i] + self.decay * self.fnn.weights[i] * self.lr
            self.fnn.biases[i] -= (self.lr / (np.sqrt(self.h_biases[i]) + self.epsilon)) * d_biases[i] + self.decay * self.fnn.biases[i] * self.lr

    def Adam(self, d_weights, d_biases):
        for i in range(self.fnn.hidden_layers + 1):
            self.hm_weights[i] = self.beta1 * self.hm_weights[i] + (1 - self.beta1) * d_weights[i]
            self.hm_biases[i] = self.beta1 * self.hm_biases[i] + (1 - self.beta1) * d_biases[i]

            self.h_weights[i] = self.beta2 * self.h_weights[i] + (1 - self.beta2) * d_weights[i]**2
            self.h_biases[i] = self.beta2 * self.h_biases[i] + (1 - self.beta2) * d_biases[i]**2

            self.hm_weights_hat = self.hm_weights[i] / (1 - self.beta1**(self.t + 1))
            self.hm_biases_hat = self.hm_biases[i] / (1 - self.beta1**(self.t + 1))

            self.h_weights_hat = self.h_weights[i] / (1 - self.beta2**(self.t + 1))
            self.h_biases_hat = self.h_biases[i] / (1 - self.beta2**(self.t + 1))

            self.fnn.weights[i] -= self.lr * (self.hm_weights_hat / ((np.sqrt(self.h_weights_hat)) + self.epsilon)) + self.decay * self.fnn.weights[i] * self.lr
            self.nn.biases[i] -= self.lr * (self.hm_biases_hat / ((np.sqrt(self.h_biases_hat)) + self.epsilon)) + self.decay * self.fnn.biases[i] * self.lr

    def NAdam(self, d_weights, d_biases):
        for i in range(self.fnn.hidden_layers + 1):
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

            self.fnn.weights[i] -= self.lr * (temp_update_w / ((np.sqrt(self.h_weights_hat)) + self.epsilon)) + self.decay * self.fnn.weights[i] * self.lr
            self.fnn.biases[i] -= self.lr * (temp_update_b / ((np.sqrt(self.h_biases_hat)) + self.epsilon)) + self.decay * self.fnn.biases[i] * self.lr


# data loader function
def load_data(type, dataset=DATASET):
    x, y, x_test, y_test = None, None, None, None
    
    if dataset == 'fashion_mnist':
        (x, y), (x_test, y_test) = fashion_mnist.load_data()

    if type == 'train':
        x_train = x.reshape(x.shape[0], 784) / 255
        y_train = np.eye(10)[y]
        return x_train, y_train
    elif type == 'test':
        x_test = x_test.reshape(x_test.shape[0], 784) / 255
        y_test = np.eye(10)[y_test]
        return x_test, y_test
    

sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize',
        'name': 'val_accuracy'
    },
    'parameters': {
        'batch_size': {
            'values': [16, 32, 64]
        }, 
        'learning_rate': {
            'values': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
        },
        'neurons': {
            'values': [16, 32, 64, 128, 256]
        },
        'hidden_layers': {
            'values': [3, 4, 5]
        },
        'activation': {
            'values': ['sigmoid', 'tanh', 'relu']
        },
        'weight_init': {
            'values': ['random', 'xavier']
        },
        'optimizer': {
            'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']
        },
        'momentum': {
            'values': [0.7, 0.8, 0.9]
        },
        'input_size': {
            'value': 784
        },
        'output_size': {
            'value': 10
        },
        'loss': {
            'value': 'cross_entropy'
        },
        'epochs': {
            'value': 10
        },
        'beta1': {
            'value': 0.9
        },
        'beta2': {
            'value': 0.999
        },
        'output_activation': {
            'value': 'softmax'
        },
        'epsilon': {
            'value': 1e-8
        },
        'decay': {
            'values': [0, 0.5, 0.0005]
        },
        'dataset': {
            'value': 'fashion_mnist'
        }
    }
}

def train_sweep():
    run = wandb.init()
    parameters = wandb.config
    run.name = f"hl={parameters['hidden_layers']}_bs={parameters['batch_size']}_ac={parameters['activation']}_neurons={parameters['neurons']}_opt={parameters['optimizer']}_mom = {parameters['momentum']}_lr={parameters['learning_rate']}_init={parameters['weight_init']}"
    x_train, y_train = load_data('train', dataset=parameters['dataset'])
    
    fnn = FeedforwardNeuralNetwork(input_size=parameters['input_size'], 
                         hidden_layers=parameters['hidden_layers'], 
                         neurons=parameters['neurons'], 
                         output_size=parameters['output_size'], 
                         acti_func=parameters['activation'], 
                         output_acti_func=parameters['output_activation'],
                         weight_init=parameters['weight_init'])
    bp = Backpropagation(fnn=fnn, 
                         loss=parameters['loss'],
                         acti_func=parameters['activation'])
    opt = Optimizer(fnn=fnn,
                    bp=bp,
                    lr=parameters['learning_rate'],
                    optimizer=parameters['optimizer'],
                    momentum=parameters['momentum'],
                    epsilon=parameters['epsilon'],
                    beta1=parameters['beta1'],
                    beta2=parameters['beta2'],
                    decay=parameters['decay'])
    
    batch_size = parameters['batch_size']
    x_train_act, x_val, y_train_act, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    print("Initial Accuracy: {}".format(np.sum(np.argmax(fnn.forward(x_train), axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]))

    for epoch in range(parameters['epochs']):
        for i in range(0, x_train_act.shape[0], batch_size):
            x_batch = x_train_act[i:i+batch_size]
            y_batch = y_train_act[i:i+batch_size]

            y_pred = fnn.forward(x_batch)
            d_weights, d_biases = bp.backward(y_batch, y_pred)
            opt.run(d_weights, d_biases)
        
        opt.t += 1

        y_pred = fnn.forward(x_train_act)
        print("Epoch: {}, Loss: {}".format(epoch + 1, loss(parameters['loss'], y_train_act, y_pred)))
        print("Accuracy: {}".format(np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_train_act, axis=1)) / y_train_act.shape[0]))

        train_loss = loss(parameters['loss'], y_train_act, y_pred)
        train_accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_train_act, axis=1)) / y_train_act.shape[0]
        val_loss = loss(parameters['loss'], y_val, fnn.forward(x_val))
        val_accuracy = np.sum(np.argmax(fnn.forward(x_val), axis=1) == np.argmax(y_val, axis=1)) / y_val.shape[0]

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })
    
    x_test, y_test = load_data('test', dataset=parameters['dataset'])
    test_loss = loss(parameters['loss'], y_test, fnn.forward(x_test))
    test_accuracy = np.sum(np.argmax(fnn.forward(x_test), axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("Test Accuracy: {}".format(test_accuracy))
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })
    
    return fnn

wandb_id = wandb.sweep(sweep_configuration, project="DA6401 - Assignment 1")

wandb.agent(wandb_id, function=train_sweep, count=20)

wandb.finish()
