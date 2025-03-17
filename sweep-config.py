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
    run.name = f"{parameters['activation']}_neurons={parameters['neurons']}_layers={parameters['hidden_layers']}_lr={parameters['learning_rate']}_batch={parameters['batch_size']}_opt={parameters['optimizer']}_mom={parameters['momentum']}_init={parameters['weight_init']}"
    x_train, y_train = load_data('train', dataset=parameters['dataset'])
    
    nn = FeedforwardNeuralNetwork(input_size=parameters['input_size'], 
                         hidden_layers=parameters['hidden_layers'], 
                         neurons=parameters['neurons'], 
                         output_size=parameters['output_size'], 
                         acti_func=parameters['activation'], 
                         output_acti_func=parameters['output_activation'],
                         weight_init=parameters['weight_init'])
    bp = Backpropagation(nn=nn, 
                         loss=parameters['loss'],
                         acti_func=parameters['activation'])
    opt = Optimizer(nn=nn,
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

    print("Initial Accuracy: {}".format(np.sum(np.argmax(nn.forward(x_train), axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]))

    for epoch in range(parameters['epochs']):
        for i in range(0, x_train_act.shape[0], batch_size):
            x_batch = x_train_act[i:i+batch_size]
            y_batch = y_train_act[i:i+batch_size]

            y_pred = nn.forward(x_batch)
            d_weights, d_biases = bp.backward(y_batch, y_pred)
            opt.run(d_weights, d_biases)
        
        opt.t += 1

        y_pred = nn.forward(x_train_act)
        print("Epoch: {}, Loss: {}".format(epoch + 1, loss(parameters['loss'], y_train_act, y_pred)))
        print("Accuracy: {}".format(np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_train_act, axis=1)) / y_train_act.shape[0]))

        train_loss = loss(parameters['loss'], y_train_act, y_pred)
        train_accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_train_act, axis=1)) / y_train_act.shape[0]
        val_loss = loss(parameters['loss'], y_val, nn.forward(x_val))
        val_accuracy = np.sum(np.argmax(nn.forward(x_val), axis=1) == np.argmax(y_val, axis=1)) / y_val.shape[0]

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })
    
    x_test, y_test = load_data('test', dataset=parameters['dataset'])
    test_loss = loss(parameters['loss'], y_test, nn.forward(x_test))
    test_accuracy = np.sum(np.argmax(nn.forward(x_test), axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("Test Accuracy: {}".format(test_accuracy))
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })
    
    return nn

wandb_id = wandb.sweep(sweep_configuration, project="CUSTOM_SWEEP")

wandb.agent(wandb_id, function=train_sweep, count=20)

wandb.finish()
