import numpy as np
import pdb

import keras
import tensorflow as tf

from keras import backend
from keras.datasets import cifar10
from keras.utils import np_utils

from cleverhans.utils_mnist import data_mnist

def data_cifar10():

    # These values are specific to CIFAR10
    img_rows = 32
    img_cols = 32
    nb_classes = 10

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    if keras.backend.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test


def main():

    train_start = 0
    train_end = 60000
    test_start = 0
    test_end = 10000

    X_train, Y_train, X_test, Y_test = data_cifar10()

    sigmas = np.arange(0.05, 0.55, 0.05)

    for sigma in sigmas:

        # Add gaussian noise
        noise_to_add = np.random.randn(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3])
        X_pert = X_test + (noise_to_add * sigma)
	X_pert = X_pert.clip(0., 1.)

        np.save("./examples/gaussian/cifar_gaussian_adv_x_" + str(sigma), X_pert)    

    np.save("./examples/gaussian/cifar_gaussian_adv_x_original", X_test)
    np.save("./examples/gaussian/cifar_gaussian_adv_y", Y_test)
   

if __name__ == "__main__":

    main()

