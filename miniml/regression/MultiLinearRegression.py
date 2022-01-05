import numpy as np
import copy
import math


class Regression():
    def __init__(self, n_iterations, learning_rate):
        #self.coefficients = None
        #self.intercept = None
        #self.gradient_descent = False
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        

    
        

    def fit(self, X, y):
        # X = self._transform_x(X)
        # y = self._transform_y(y)

        # betas = self._estimate_coefficients(X, y)

        # self.intercept = betas[0]

        # self.coefficients = betas[1:]
        n_samples, n_features = X.shape
        #X = np.insert(X, 0, 1, axis=1)
        self.weights = np.random.uniform(-1 / math.sqrt(n_features), 1 / math.sqrt(n_features), (n_features, ))
        self.bias = 0

        for i in range(self.n_iterations):
            y_hat = np.dot(X, self.weights) + self.bias # y_hat = X*W+b

            dw = (1 / n_samples) * np.dot(X.T, (y_hat - y))
            db = (1 / n_samples) * np.sum(y_hat - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        y = b_0 + b_1*x + ... + b_i*x_i
        """
        print(X)
        #X = np.insert(X, 0, 1, axis=1)
        print(X)
        y_approximated = np.dot(X, self.weights)
        return y_approximated

class LinearRegression(Regression):
    def __init__(self, n_iterations=100, learning_rate=0.001, gradient_descent=True):
        self.gradient_descent = gradient_descent
        super(LinearRegression, self).__init__(n_iterations=n_iterations,
                                            learning_rate=learning_rate)
    def fit(self, X, y):
        # If not gradient descent => Least squares approximation of w
        if not self.gradient_descent:
            '''
            β = (X^T X)^-1 X^T y
            Estimates both the intercept and all coefficients.
            '''
            X_T = X.transpose()

            inversed = np.linalg.inv( X_T.dot(X) )
            self.weights = inversed.dot(X_T).dot(y)
            print(self.weights)
        else:
            super(LinearRegression, self).fit(X, y)
