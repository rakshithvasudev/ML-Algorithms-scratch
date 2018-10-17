import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class RidgeRegression:
    """
    This class is responsible for building a ridge regressor with penalization.
    The objective function is to minimize the effect of mean of least squares and effective sum of
    product of the lambda - regularization constant and the squared norm of the weights.
    i.e min sigma(i=1 to n)[(yi-Xi.T*w)^2 +lambda*||w||^2]
    """
    def __init__(self, lambdA=0.1):
        """
        Constructor to initalize the regressor.
        :param lambdA: the penalization constant on the training examples. Larger the value,
        larger the effect of regularization.
        """
        self.lambdA = lambdA
        self.w = None

    def fit(self, X, y):
        """
        X: Numpy training matrix like [[feature1, feature2,...feature p],.....
                                      [feature1, feature2,...feature p]].
        y: vector of outputs like [[target1.....targetp]]

        fits the training examples based on the formula. It is important to maintain the
        """

        # the matrix form of the ridge objective function is
        # w = [((X.T * X + lambda*I)^-1) *(X.T*y)] where I is the Identity matrix of dimensions(m*m).
        # where m is the number of features in X.
        self.w = np.linalg.inv(X.T.dot(X) + self.lambdA * np.eye(X.shape[1])).dot(X.T.dot(y))
        print("RidgeRegression(lambdA={})".format(self.lambdA))

    def predict(self, X):
        """
        predict the output from the learnt weight matrix.
        :param X: observations in the test set like  [[feature1, feature2,...feature p],.....
                                                    [feature1, feature2,...feature p]].
        :return:  predicted target values.
        """
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

    print("Input for Ridge regression :{}".format(X_test))
    print("Ridge regression Predictions :{}".format(r_preds))
    print("MSE: {}".format(mean_squared_error(y_test, r_preds)))
