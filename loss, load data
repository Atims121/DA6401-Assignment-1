def loss(loss, y, y_pred):
    if loss == "cross_entropy": # Cross Entropy
        return -np.sum(y * np.log(y_pred))
    elif loss == "mean_squared_error": # Mean Squared Error
        return np.sum((y - y_pred) ** 2) / 2
    else:
        raise Exception("Invalid loss function")


def load_data(type, dataset=DATASET):
    x, y, x_test, y_test = None, None, None, None
    
    if dataset == 'mnist':
        (x, y), (x_test, y_test) = mnist.load_data()
    elif dataset == 'fashion_mnist':
        (x, y), (x_test, y_test) = fashion_mnist.load_data()

    if type == 'train':
        x_train = x.reshape(x.shape[0], 784) / 255
        y_train = np.eye(10)[y]
        return x_train, y_train
    elif type == 'test':
        x_test = x_test.reshape(x_test.shape[0], 784) / 255
        y_test = np.eye(10)[y_test]
        return x_test, y_test
