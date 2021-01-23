import numpy as np
from sklearn.datasets import make_regression
from tiny_ml.utils import train_test_split
from tiny_ml import LinearRegression
import matplotlib.pyplot as plt


def main():
    X, y = make_regression(n_samples=100, n_features=1, noise=30)
    X_train, X_test, y_train, y_test = train_test_split(X, y)


    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = (np.mean(y_test - y_pred)**2)
    # Print MSE
    print ('Mean Squared Error: %.4f' % mse)
    y_pred_line = model.predict(X)
    X = X[:, 1]
    X_train = X_train[:, 1]
    X_test = X_test[:, 1]

    # Color map
    cmap = plt.get_cmap('viridis')

    # Plot the results
    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(366 * X, y_pred_line, color='black', linewidth=2, label="Prediction")
    plt.suptitle("Linear Regression")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()