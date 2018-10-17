import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class RidgeRegression:
    """
    This class is responsible for building a ridge regressor with penalization.

    """
    def __init__(self, lambdA=0.1):
        """
        Constructor to initalize the regressor.
        :param lambdA: the penalization constant
        """
        self.lambdA = lambdA
        self.w = None

    def fit(self, X, y):
        """
        X: Numpy training matrix
        y: vector of outputs
        fits the training examples based on the formula.
        """

        self.w = np.linalg.inv(X.T.dot(X) + self.lambdA * np.eye(X.shape[1])).dot(X.T.dot(y))
        print("RidgeRegression(lambdA={})".format(self.lambdA))

    def predict(self, X):
        return X.dot(self.w)

    def getlambdA(self):
        return self.lambdA

    def __str__(self):
        return "Lambda: {}".format(str(self.lambdA))


if __name__ == '__main__':
    ridge_regressor = RidgeRegression(2)
    content = np.loadtxt("winequality-white.csv", delimiter=";", skiprows=1)
    X = content[:, :-1]
    y = content[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    ridge_regressor.fit(X_train, y_train)
    r_preds = ridge_regressor.predict(X_test)
    print(mean_squared_error(y_test, r_preds))
