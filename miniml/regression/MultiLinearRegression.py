import numpy as np
import copy


class MultiLinearRegression():
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        X = self._transform_x(X)
        y = self._transform_y(y)

        betas = self._estimate_coefficients(X, y)

        self.intercept = betas[0]

        self.coefficients = betas[1:]

    def predict(self, X):
        """
        y = b_0 + b_1*x + ... + b_i*x_i
        """
        predictions = []
        for index, row in X.iterrows():
            values = row.values
            pred = np.multiply(values, self.coefficients) # multiply argumets element-wise
            pred = sum(pred)
            pred += self.intercept

            predictions.append(pred)

        return predictions

    def r2_score(self, y_true, y_pred):
        '''
            r2 = 1 - (rss/tss)
            rss = sum_{i=0}^{n} (y_i - y_hat)^2
            tss = sum_{i=0}^{n} (y_i - y_bar)^2
        '''
        y_values = y_true.values()
        y_average = np.average(y_values)

        residual_sum_of_square = 0
        total_sum_of_squares = 0

        for i in range(len(y_values)):
            residual_sum_of_squares += (y_values[i] - y_pred[i]) ** 2
            total_sum_of_squares += (y_values[i] - y_average[i]) ** 2

        return 1 - (residual_sum_of_square / total_sum_of_squares)

    def _transform_x(self, X):
        X = copy.deepcopy(X)
        X.insert(0, 'ones', np.ones((X.shape[0], 1)))
        return X.values
    
    def _transform_y(self, y):
        y = copy.deepcopy(y)
        return y.values

    def _estimate_coefficients(self, X, y):
        '''
            β = (X^T X)^-1 X^T y
            Estimates both the intercept and all coefficients.
        '''
        X_T = X.transpose()

        inversed = np.linalg.inv( X_T.dot(X) )
        coefficients = inversed.dot(X_T).dot(y)

        return coefficients
