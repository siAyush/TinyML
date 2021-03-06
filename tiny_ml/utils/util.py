import numpy as np
import matplotlib.pyplot as plt


class Plot():
    def __init__(self):
        self.cmap = plt.get_cmap('plasma')

    def _transform(self, X, dim):
        e_val, e_vec = np.linalg.eig(np.cov(X, rowvar=False))
        idx = e_val.argsort()[::-1]
        e_vec = e_vec[:, idx][:, :dim]
        return X.dot(e_vec)

    # Plot the dataset X and the corresponding labels y in 2D using PCA.
    def plot_in_2d(self, X, y=None, title=None, accuracy=None, legend_labels=None):
        X_transformed = self._transform(X, dim=2)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]

        class_distr = []
        y = np.array(y).astype(int)
        colors = [self.cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]
        for i, l in enumerate(np.unique(y)):
            _x1 = x1[y == l]
            _x2 = x2[y == l]
            _y = y[y == l]
            class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))

        # Plot legend
        if not legend_labels is None:
            plt.legend(class_distr, legend_labels, loc=1)

        # Plot title
        if title:
            if accuracy:
                perc = 100 * accuracy
                plt.suptitle(title)
                plt.title("Accuracy: %.1f%%" % perc, fontsize=10)
            else:
                plt.title(title)

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()


def shuffle_data(x, y, seed=None):
    if seed:
        np.random.seed(seed)
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    return x[idx], y[idx]


def train_test_split(x, y, test_size=0.3, shuffle=True, seed=None):
    if shuffle:
        x, y = shuffle_data(x, y, seed)
    split_i = len(y) - int(len(y) // (1 / test_size))
    x_train, x_test = x[:split_i], x[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]
    return x_train, x_test, y_train, y_test


def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)