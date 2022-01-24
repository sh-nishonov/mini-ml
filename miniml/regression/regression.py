import numpy as np
import math


class Regression:
    def __init__(
        self,
        n_iterations,
        learning_rate,
        l1_regularization=False,
        l2_regularization=False,
        alpha=0,
    ):

        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.l1_regularization = l1_regularization
        self.alpha = alpha
        self.l2_regularization = l2_regularization

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.random.uniform(
            -1 / math.sqrt(n_features), 1 / math.sqrt(n_features), (n_features,)
        )
        self.bias = 0

        for _ in range(self.n_iterations):
            y_hat = np.dot(X, self.weights) + self.bias  # y_hat = X*W+b
            if self.l1_regularization:
                dw = (1 / n_samples) * np.dot(X.T, (y_hat - y)) + self.alpha * np.sign(
                    self.weights
                )
            elif self.l2_regularization:
                dw = (1 / n_samples) * np.dot(
                    X.T, (y_hat - y)
                ) + self.alpha * self.weights
            else:
                dw = (1 / n_samples) * np.dot(X.T, (y_hat - y))
            db = (1 / n_samples) * np.sum(y_hat - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        y = b_0 + b_1*x + ... + b_i*x_i
        """
        y_approximated = np.dot(X, self.weights)
        return y_approximated


class LinearRegression(Regression):
    def __init__(self, n_iterations=100, learning_rate=0.001, gradient_descent=True):
        self.gradient_descent = gradient_descent
        super().__init__(n_iterations=n_iterations, learning_rate=learning_rate)

    def fit(self, X, y):
        # If not gradient descent => Least squares approximation of w
        if not self.gradient_descent:
            """
            β = (X^T X)^-1 X^T y
            Estimates both the intercept and all coefficients.
            """

            XtX = np.dot(X.T, X)
            inversed = np.linalg.inv(XtX)
            Xty = np.dot(X.T, y)
            self.weights = inversed.dot(Xty)

        else:
            super(LinearRegression, self).fit(X, y)


class RidgeRegression(Regression):
    def __init__(self, alpha, n_iterations=None, learning_rate=None):

        super().__init__(
            n_iterations,
            learning_rate,
            l1_regularization=False,
            l2_regularization=True,
            alpha=alpha,
        )

    def fit(self, X, y):
        """β=(X^T * X+λI_prime)^(-1)X^Ty."""
        XtX = np.dot(X.T, X)
        I_prime = np.eye(
            X.shape[1]
        )  # creates Identity matrix with the size of (X.shape[1], X.shape[1])
        I_prime[0, 0] = 0
        XtX_alpha_inverse = np.linalg.inv(XtX + self.alpha * I_prime)
        Xty = np.dot(X.T, y)
        self.weights = np.dot(XtX_alpha_inverse, Xty)


class LassoRegression(Regression):
    def __init__(self, n_iterations, learning_rate, alpha):
        super().__init__(
            n_iterations,
            learning_rate,
            l1_regularization=True,
            l2_regularization=False,
            alpha=alpha,
        )
