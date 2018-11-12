import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class LinearRegressor:
    """
    This class is responsible for building a linear regressor.
    The objective function is to minimize the effect of mean of least squares
    i.e min sigma(i=1 to n)[(yi-Xi.T*w)^2]
    """
    def __init__(self):
            """
            Constructor to initalize the regressor.
            """
            self.weights = None

    def fit(self, X, y):
        """
        X: Numpy training matrix like [[feature1, feature2,...feature p],.....
                                      [feature1, feature2,...feature p]].
        y: vector of outputs like [[target1.....targetp]]

        fits the training examples based on the formula. It is important to maintain the order.
        """

        # the matrix form of the ridge objective function is
        # w = [((X.T * X)*(X.T*Y)].
        # where m is the number of features in X.
        print("Linear regressor())")
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))

    def predict(self, X):
        """
        predict the output from the learnt weight matrix.
        :param X: observations in the test set like  [[feature1, feature2,...feature p],.....
                                                    [feature1, feature2,...feature p]].
        :return:  predicted target values.
        """
        return X.dot(self.w)



if __name__ == '__main__':
    linear_regressor = LinearRegressor()
    content = np.loadtxt("winequality-white.csv", delimiter=";", skiprows=1)

    X = content[:, :-1]
    y = content[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    linear_regressor.fit(X_train, y_train)

    r_preds = linear_regressor.predict(X_test)

    print("Input for Ridge regression :{}".format(X_test))
    print("Linear regression Predictions :{}".format(r_preds))
    print("MSE: {}".format(mean_squared_error(y_test, r_preds)))
