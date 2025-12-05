from data_loader import MNISTLoader
from PySide6.QtWidgets import QApplication
from neural_network import MLP
from painter import DigitApp
import sys
import numpy as np

def accuracy(y, pred):
    return np.mean(y == pred)

def train(model, X, y, epochs=10, lr=0.01, batch_size=64):
    n = len(y)
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        X, y = X[perm], y[perm]
        losses = []
        for i in range(0, n, batch_size):
            xb, yb = X[i:i+batch_size], y[i:i+batch_size]
            probs = model.forward(xb)
            clipped = np.clip(probs, 1e-12, 1.0)
            loss = -np.mean(np.log(clipped[np.arange(len(yb)), yb]))
            losses.append(loss)
            model.backward(yb, lr)
        avg_loss = np.mean(losses)
        preds = model.predict(X)
        acc = accuracy(y, preds)
        if epoch % 10 == 0:
            print(f"[Epoch {epoch}/{epochs}] loss={avg_loss:.4f} | acc={acc:.4f}")

def main():
    loader = MNISTLoader()
    X_train, y_train, X_test, y_test = loader.load()

    model = MLP([784, 64, 64, 10], activation="relu")

    train(model, X_train, y_train, epochs=100, lr=0.01, batch_size=64)

    app = QApplication(sys.argv)
    win = DigitApp(model)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
