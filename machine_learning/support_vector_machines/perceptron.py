import numpy as np


class Perceptron:
    def __init__(self, n_in: int) -> None:
        self.w = np.random.rand(n_in)
        self.b = np.random.rand(1)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            lr: float=0.1) -> None:
        errors = 1
        while errors > 0:
            errors = 0
            for i in range(X.shape[0]):
                z = np.dot(self.w.T, X[i]) + self.b
                y_hat = np.sign(z)
                if y[i] != y_hat:
                    errors += 1
                    dw, db = -y[i]*X[i], -y[i]
                    self.w = self.w - lr * dw
                    self.b = self.b - lr * db