import numpy as np
import pdb

import tensorflow as tf

from cleverhans.utils_mnist import data_mnist

def main():

    train_start = 0
    train_end = 60000
    test_start = 0
    test_end = 10000

    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Use label smoothing
    assert Y_train.shape[1] == 10
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    sigmas = np.arange(0.05, 0.55, 0.05)

    for sigma in sigmas:

        # Add gaussian noise
        noise_to_add = np.random.randn(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3])
        X_pert = X_test + (noise_to_add * sigma)
	X_pert = X_pert.clip(0., 1.)

        np.save("./examples/gaussian/mnist_gaussian_adv_x_" + str(sigma), X_pert)    
    
    np.save("./examples/gaussian/mnist_gaussian_adv_y", Y_test)
   

if __name__ == "__main__":

    main()

