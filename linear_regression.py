import numpy as np

# initialize the init
class LinearRegression:
    def __init__(self):
        self.w = None
        self.b = None

    # create the fit method
    def fit(self, X, y):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("Input data must be numpy arrays.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Dimensions of X and y are not compatible.")

        # add a column of ones to X for the intercept term
        X = np.c_[X, np.ones(X.shape[0])]

        # calculate the b and w parameters
        X_transpose = X.T
        self.w = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
        self.b = self.w[-1]  # Last element of the weight vector is the intercept

    # create the predict method
    def predict(self, X):
        if self.w is None:
            raise ValueError("Model has not been trained.")
        X = np.c_[X, np.ones(X.shape[0])]
        return X.dot(self.w)

    # create the evaluate method
    def evaluate(self, X, y):
        if self.w is None:
            raise ValueError("Model has not been trained.")
        # call the predict method here so we don't need to call it seperatly later
        y_pred = self.predict(X)
        mse = np.mean((y_pred - y) ** 2)
        return y_pred, mse

