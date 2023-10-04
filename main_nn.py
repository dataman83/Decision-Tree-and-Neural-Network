import numpy as np
import matplotlib.pyplot as plt
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10():
    train_batches = []
    for i in range(1, 6):
        batch = unpickle(f"cifar-10-batches-py/data_batch_{i}")
        train_batches.append((batch[b'data'], batch[b'labels']))

    test_batch = unpickle("cifar-10-batches-py/test_batch")

    X_train = np.vstack([batch[0] for batch in train_batches]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.hstack([batch[1] for batch in train_batches])

    X_test = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(test_batch[b'labels'])   # convert list to numpy array

    return X_train, y_train, X_test, y_test


# Normalize input data
def normalize_data(X_train, X_test):
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, X_test

# Convert labels to one-hot encoding
def one_hot_encoding(y_train, y_test):
    num_classes = np.max(y_train) + 1
    y_train_encoded = np.zeros((y_train.shape[0], num_classes))
    y_train_encoded[np.arange(y_train.shape[0]), y_train] = 1
    y_test_encoded = np.zeros((y_test.shape[0], num_classes))
    y_test_encoded[np.arange(y_test.shape[0]), y_test] = 1
    return y_train_encoded, y_train, y_test_encoded, y_test  # Return original labels as well

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Softmax activation function
def softmax(x):
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


# Neural network architecture
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.probs = softmax(self.z2)
        return self.probs

    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        delta3 = self.probs
        delta3[np.arange(m), np.argmax(y, axis=1)] -= 1
        delta3 /= m

        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)

        delta2 = np.dot(delta3, self.W2.T) * sigmoid_prime(self.z1)

        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def calculate_loss(self, X, y_encoded):
        N = X.shape[0]
        probs = self.forward(X)
        corect_logprobs = -np.log(probs) * y_encoded
        data_loss = np.sum(corect_logprobs)
        return 1. / N * data_loss

    # Accuracy calculation
    def accuracy(self, X, y_encoded):
        probs = self.forward(X)
        y_pred = np.argmax(probs, axis=1)
        y_true = np.argmax(y_encoded, axis=1)
        return np.mean(y_pred == y_true)


# Training the model
def train_model(X_train, y_train_encoded, y_train, X_val, y_val_encoded, y_val, learning_rate, epochs, batch_size, input_dim, hidden_dim, output_dim):
    model = NeuralNetwork(input_dim, hidden_dim, output_dim)
    loss_history = []
    accuracy_history = []

    for i in range(epochs):
        for j in range(0, X_train.shape[0], batch_size):
            batch_X = X_train[j:j + batch_size]
            batch_y = y_train[j:j + batch_size]
            batch_y_encoded = y_train_encoded[j:j + batch_size]

            model.forward(batch_X)
            model.backward(batch_X, batch_y_encoded, learning_rate)

        # Calculate loss and accuracy at the end of each epoch
        loss = model.calculate_loss(X_val, y_val_encoded)  # Pass one-hot encoded labels
        accuracy = model.accuracy(X_val, y_val_encoded)  # Pass one-hot encoded labels to predict function

        loss_history.append(loss)
        accuracy_history.append(accuracy)
        print(f"Epoch {i + 1}/{epochs}: Training accuracy: {model.accuracy(X_train, y_train_encoded):.4f}, Test accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

    # Print final weights of the network
    print("Final weights of the network:")
    print("W1:", model.W1)
    print("b1:", model.b1)
    print("W2:", model.W2)
    print("b2:", model.b2)

    return loss_history, accuracy_history

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_cifar10()
    X_train, X_test = normalize_data(X_train, X_test)
    y_train_encoded, y_train, y_test_encoded, y_test = one_hot_encoding(y_train, y_test)

    input_dim = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
    output_dim = y_train_encoded.shape[1]

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    learning_rates = [0.001, 0.01, 0.1, 1.0, 10, 100]
    batch_sizes = [1, 5, 20, 100, 300]

    max_acc_lr = {}  # maximum accuracies for each learning rate
    max_acc_bs = {}  # maximum accuracies for each batch size


    # 1. Train a model with given parameters and plot test accuracy vs epoch
    loss_history, accuracy_history = train_model(X_train, y_train_encoded, y_train, X_test, y_test_encoded, y_test, 0.1, 20, 100, input_dim, 30, output_dim)
    plt.plot(accuracy_history)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.show()
    print("Maximum accuracy achieved:", max(accuracy_history))

    # 2. Train models with different learning rates
    for lr in learning_rates:
        loss_history, accuracy_history = train_model(X_train, y_train_encoded, y_train, X_test, y_test_encoded, y_test, lr, 20, 100, input_dim, 30, output_dim)
        max_acc_lr[lr] = max(accuracy_history)
        plt.plot(accuracy_history, label=f'lr={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.show()
    print("Maximum accuracy achieved for each learning rate:", max_acc_lr)

    # 3. Train models with different batch sizes
    for bs in batch_sizes:
        loss_history, accuracy_history = train_model(X_train, y_train_encoded, y_train, X_test, y_test_encoded, y_test, 0.1, 20, bs, input_dim, 30, output_dim)
        max_acc_bs[bs] = max(accuracy_history)
    plt.plot(list(max_acc_bs.keys()), list(max_acc_bs.values()))
    plt.xlabel('Batch Size')
    plt.ylabel('Max Test Accuracy')
    plt.show()
    print("Maximum accuracy achieved for each batch size:", max_acc_bs)


