# Yael Livne 313547929, Nir David 313231805
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import numpy as np
import matplotlib.pyplot as plt
import time

ETTA = 0.2
EPOCHES = 6
MAX_GD = 10e5  # maximum time for GD
MIN_ACC_EVOLUTION = 1e-5  # min for accuracy change


def get_data_set():
    """
    Help function from Moodle guide, to import the MNIST dB
    :return:
    x train and test sets
    y labels - train and test
    """

    start_import_time = time.time()
    print("Importing MNIST dB started ")
    mnist = fetch_openml('mnist_784')
    x = mnist['data'].astype('float32')
    y = mnist['target']
    random_state = check_random_state(0)
    permutation = random_state.permutation(x.shape[0])
    x = x[permutation]
    y = y[permutation]  # The next line flattens the vector into 1D array of size 784
    x = x.reshape((x.shape[0], -1))
    x_train_pre, x_test_pre, y_train_pre, y_test_pre = train_test_split(x, y, test_size=0.2)
    scaler = StandardScaler()  # the next lines standardize the images
    x_train_pre = scaler.fit_transform(x_train_pre)
    x_test_pre = scaler.transform(x_test_pre)
    time_finished_importing = int(time.time() - start_import_time)
    print(f"Importing MNIST dB finished in {time_finished_importing} sec")
    return x_train_pre, x_test_pre, np.array(y_train_pre, dtype=np.float64), np.array(y_test_pre, dtype=np.float64)


def to_categorical(y, dtype='float32'):
    """
    Parameters
    ----------
    y : np array float64
        original vector.
    dtype :'float64', optional

    Returns
    -------
    categorical : array of float32
        The digits 0 through 9 are represented as a set of nine zeros and a single one.
        The digit is determined by the location of the number 1.
    """
    y = np.array(y, dtype='int')
    num_classes = 10
    input_shape = y.shape
    y = y.ravel()
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def add_one_last(x):
    """

    Parameters
    ----------
    x : np.array(float64)
        Matrix of shape(a,b)

    Returns
    -------
    a : np.array(float64)
        Matrix of shape(a,b+1) where the last column is all ones and the rest is the original matrix x.

    """
    # adds last column of 1s to matrix x
    shape_x = list(x.shape)
    shape_x[1] += 1
    a = np.ones(shape_x)
    a[:, :-1] = x
    return a


def create_initial_w(shape, sigma):
    """

    Parameters
    ----------
    shape : tuple
        Shape of the w matrix.
    sigma : float
        sigma value for gaussian variance.

    Returns
    -------
    wt : np.array(float64)
        initial weights matrix.

    """
    # create a matrix of initial weights- each a Gaussian RV  0-mean sigma-variance
    wt = np.random.normal(0, sigma, shape)
    wt = add_one_last(wt)
    return wt


def pre_training_data_modification(x_train_pre, x_test_re, y_train_pre, y_test_pre):
    """
    Returns
    -------
    x_train_with_bias.T : np.array(float64)
        x train set vector with bias.
    x_test_with_bias.T : np.array(float64)
        x test set vector with bias.
    y_train_categorical.T : np.array(float32)
        one hot representation of the y train set.
    y_test_categorical.T : np.array(float32)
        one hot representation of the y test set.
    y_test : np.array(float64)
        test set labels (original as was imported).
    y_train : np.array(float64)
        train set labels (original as was imported).

    """
    pre_order_start_time = time.time()
    print("started pre training data modification")
    # x_train = np.load('x_train.npy')
    # x_test = np.load('x_test.npy')
    # y_train = np.load('y_train.npy')
    # y_test = np.load('y_test.npy')

    # hot representation of train and test labels (y)
    y_train_categorical = to_categorical(y_train_pre)
    y_test_categorical = to_categorical(y_test_pre)
    # add bias to x
    x_train_with_bias = add_one_last(x_train_pre)
    x_test_with_bias = add_one_last(x_test_re)
    time_finished = time.time() - pre_order_start_time
    print(f"pre training data modification ended in {time_finished:.3f} sec")
    return x_train_with_bias.T, x_test_with_bias.T, y_train_categorical.T, y_test_categorical.T, y_test_pre, y_train_pre


def train_model(x_train_set, x_test_set, y_train_set, y_test_set, y_test_labels, y_train_labels):
    """
    Parameters
    ----------
    x_train_set : np.array(float64)
        train set images, with bias.
    x_test_set : np.array(float64)
        test set images, with bias.
    y_train_set : np.array(float32)
        one hot representation of train set.
    y_test_set : np.array(float32)
        one hot representation of test set.
    y_test_labels : np.array(float64)
        original labels of test set.
    y_train_labels : np.array(float64)
        original labels of train set.
    ----------
    Training the model by:
    1. apply softmax on the train set - call softmax()
    2. using gradient descent for each iteration, we update:
        wj(r+1) = wj(r)-etta*err_function
    3. find accuracy of each iteration - call calc_accuracy()
    Returns
    -------
    None.

    """

    # Initialize
    print("Started training model")
    iteration, acc, delta_acc = 0, 0, 100
    start_train_time = time.time()
    w_transpose = create_initial_w((10, 784), 1 / 1000)

    for epoch in range(1, EPOCHES):
        while delta_acc >= MIN_ACC_EVOLUTION and iteration <= MAX_GD:
            iteration += 1
            acc_old = acc
            # apply softmax on train set
            a_k = softmax(np.matmul(w_transpose, x_train_set))
            # Gradient Descent
            err = calc_error_function(a_k, x_train_set, y_train_set)
            w_transpose = w_transpose - ETTA * err
            # Accuracy check - using test set
            b_k = softmax(np.matmul(w_transpose, x_test_set))
            b_k_max_loc = np.zeros(b_k.shape)
            b_k_max_loc[np.argmax(b_k, axis=0), np.arange(b_k.shape[1])] = 1
            acc = calc_accuracy(b_k_max_loc, y_test_set)
            delta_acc = abs(acc_old - acc)
            print(f"Model accuracy for iteration {iteration} is- {acc:.3f} accuracy evolution is- {delta_acc:.4f}")

    print('Train finished in {} sec'.format(int(time.time() - start_train_time)))


def calc_accuracy(b_k_max_loc, y_test_data):
    """
    Parameters
    ----------
    b_k_max_loc : np.array with same shape as test set
        matrix with 1 where argmax{y_test_set}.
    y_test_data : np.array
        test set wth one hot representation.

    Returns
    -------
    float
        accuracy of the model for a given iteration.

    (#correct classifications on test set/size of test set) * 100%
    """
    return np.sum(np.logical_and(b_k_max_loc, y_test_data)) / y_test_data.shape[1]


def softmax(a):
    """
    Parameters
    ----------
    a : np.array
        input vector for softmax function.

    Returns
    -------
    soft_max : np.array
        normalized probability vector s.t its parts sums to 1.
    """
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a, axis=0)  # axis 0 is summation for columns
    soft_max = exp_a / sum_exp_a
    return soft_max


def calc_error_function(a_k, x_k, y_k):
    """
    Parameters
    ----------
    a_k : np.array(dytpe = float64)
        a_k = w^T*x.
    x_k : np.array(dytpe = float64)
        image train set with bias.
    y_k : np.array(dytpe = float64)
        one hot representation of train set.

    Returns
    -------
    np.array(dytpe = float64)
        err matrix of size (10,785).

    """
    dim = a_k.shape[1]
    return (-1 / dim) * np.matmul((y_k - a_k), x_k.T)


if __name__ == '__main__':
    start_time = time.time()
    x_train, x_test, y_train, y_test = get_data_set()
    x_train_w_bias, x_test_w_bias, y_train_one_hot, y_test_one_hot, y_test_label, y_train_label = \
        pre_training_data_modification(x_train, x_test, y_train, y_test)
    train_model(x_train_w_bias, x_test_w_bias, y_train_one_hot, y_test_one_hot, y_test_label, y_train_label)
    print(f"Program finished in {int(time.time() - start_time)}")
