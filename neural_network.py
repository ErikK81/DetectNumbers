import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return (x > 0).astype(np.float32)

class MLP:
    def __init__(self, dims, activation="relu", seed=42):
        rng = np.random.RandomState(seed)

        if activation == "relu":
            self.activation, self.d_activation = relu, d_relu

        self.layers = []
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            W = rng.randn(fan_in, dims[i+1]).astype(np.float32) * np.sqrt(2 / fan_in)
            b = np.zeros((1, dims[i+1]), dtype=np.float32)
            self.layers.append([W, b])

    def forward(self, X):
        A = X
        self.a = [A]
        self.z = []

        for W, b in self.layers[:-1]:
            Z = A @ W + b
            A = self.activation(Z)
            self.z.append(Z)
            self.a.append(A)

        # camada de saída (logits)
        W, b = self.layers[-1]
        logits = A @ W + b
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        self.a.append(probs)
        return probs

    def backward(self, y, lr=0.01):
        num_classes = self.layers[-1][0].shape[1]
        y_oh = np.eye(num_classes)[y]
        m = len(y)

        delta = self.a[-1] - y_oh

        for i in reversed(range(len(self.layers))):
            A_prev = self.a[i]
            W, b = self.layers[i]

            dW = (A_prev.T @ delta) / m
            db = delta.sum(axis=0, keepdims=True) / m

            self.layers[i][0] -= lr * dW
            self.layers[i][1] -= lr * db

            if i > 0:
                Z_prev = self.z[i-1]
                delta = (delta @ W.T) * self.d_activation(Z_prev)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def save(self, path="model.npy"):
        np.save(path, self.layers, allow_pickle=True)

    def load(self, path="model.npy"):
        self.layers = np.load(path, allow_pickle=True)
