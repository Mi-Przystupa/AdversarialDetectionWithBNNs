from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import logging
import pdb

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans_tutorials.tutorial_models import make_basic_cnn
from cleverhans.utils import AccuracyReport, set_log_level

def tutorial():

    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    sess = tf.Session()

    # Get MNIST test data
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

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # model_path = "models/mnist"
    # Train an MNIST model

    batch_size = 128

    train_params = {
        'nb_epochs': 1,
        'batch_size': batch_size,
        'learning_rate': 0.001
    }

    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}

    rng = np.random.RandomState([2017, 8, 30])

    model = make_basic_cnn(nb_filters=64)
    preds = model.get_probs(x)

    def evaluate():

        # Evaluate the accuracy of the MNIST model on legitimate test
        # examples
        eval_params = {'batch_size': batch_size, 'adversarial': False}
        acc = model_eval(
            sess, x, y, preds, X_test, Y_test, args=eval_params)
        report.clean_train_clean_eval = acc
        assert X_test.shape[0] == test_end - test_start, X_test.shape
        print('Test accuracy on legitimate examples: %0.4f' % acc)
        
    model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                args=train_params, rng=rng)

    eval_params = {'batch_size': batch_size, 'adversarial': False}
    acc = model_eval(sess, x, y, preds, X_train, Y_train, args=eval_params)

    # Initialize the Fast Gradient Sign Method (FGSM) attack object and
    fgsm = FastGradientMethod(model, sess=sess)
    adv_x = fgsm.generate(x, **fgsm_params)
    preds_adv = model.get_probs(adv_x)

    # Define adversarial examples placeholder
    adv_examples = tf.placeholder(tf.float32, [None, 28, 28, 1])

    # Evaluate the accuracy of the MNIST model on adversarial examples
    eval_par = {'batch_size': batch_size, 'adversarial': True}
    acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
    print('Test accuracy on adversarial examples: %0.4f\n' % acc)
    report.clean_train_adv_eval = acc

    # Write the adversarial examples to a file
    np_examples = adv_x.eval(session=sess, feed_dict={x : X_train})
    np.save("adv_examples", np_examples)

    pdb.set_trace()

if __name__ == "__main__":
    tutorial()
