import urllib.request, gzip, os
import numpy as np

class MNISTLoader:
    BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    FILES = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    def __init__(self, folder="mnist_data"):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)

    def _download(self, fname):
        path = os.path.join(self.folder, fname)
        if not os.path.exists(path):
            urllib.request.urlretrieve(self.BASE + fname, path)
        return path

    def _parse(self, path, is_image):
        with gzip.open(path, 'rb') as f:
            raw = f.read()
        if is_image:
            arr = np.frombuffer(raw, np.uint8, offset=16)
            return arr.reshape(-1, 28*28).astype(np.float32) / 255.0
        return np.frombuffer(raw, np.uint8, offset=8).astype(int)

    def load(self):
        X_train = self._parse(self._download(self.FILES["train_images"]), True)
        y_train = self._parse(self._download(self.FILES["train_labels"]), False)
        X_test = self._parse(self._download(self.FILES["test_images"]), True)
        y_test = self._parse(self._download(self.FILES["test_labels"]), False)
        return X_train, y_train, X_test, y_test
