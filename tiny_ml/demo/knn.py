import numpy as np
from sklearn import datasets
from tiny_ml.utils import Plot, normalize, train_test_split
from tiny_ml import KNN


def main():
    data = datasets.load_iris()
    # using two class only
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = 0
    y[y == 2] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = KNN()
    y_pred = clf.predict(5, X_test, X_train, y_train)
    accuracy = np.mean(y_pred == y_test)
    print ("Accuracy:", accuracy)

    Plot().plot_in_2d(X, y,title="Logistic Regression", accuracy=accuracy, legend_labels=data['target_names'])


if __name__ == "__main__":
    main()