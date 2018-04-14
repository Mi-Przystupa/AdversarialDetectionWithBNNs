import pdb
import os

import keras
import tensorflow as tf
import numpy as np

from tensorflow.python.platform import flags

import scipy.misc
from scipy.misc import imread, imresize
from vgg.vgg16_keras import vggbn
from vgg.imagenet_classes import class_names

from keras import backend
from keras.datasets import cifar10
from keras.utils import np_utils

from cleverhans.attacks_tf import fgsm_old
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks_tf import jacobian_graph
from cleverhans.utils import other_classes, pair_visual, grid_visual
from cleverhans.utils_tf import model_train, model_eval, batch_eval

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', os.getcwd(), 'Directory storing the saved model.')
flags.DEFINE_string('filename', 'ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')
flags.DEFINE_boolean('viz_enabled', False, 'Enable sample visualization.')
flags.DEFINE_integer('nb_classes', 10, 'Number of classification classes')
flags.DEFINE_integer('img_rows', 32, 'Input row dimension')
flags.DEFINE_integer('img_cols', 32, 'Input column dimension')
flags.DEFINE_integer('nb_channels', 3, 'Nb of color channels in the input.')
flags.DEFINE_integer('nb_filters', 64, 'Number of convolutional filter to use')
flags.DEFINE_integer('nb_pool', 2, 'Size of pooling area for max pooling')
flags.DEFINE_integer('source_samples', 50000, 'Nb of test set examples to attack')


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


def generate_images():

    print('==> Preparing data..')
    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    print "==> Beginning Session"

    # Get CIFAR10 test data
    X_train, Y_train, X_test, Y_test = data_cifar10()

    assert Y_train.shape[1] == 10.
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Load model
    args_model = 'vgg6'
    args_load = 'cifar10vgg.h5'
    args_pool = 0.05
    args_attack = 'fgsm'

    print "==> loading vgg model"
    model = vggbn(top=True, pool=args_pool)
    model.load_weights(args_load)
    predictions = model(x)

    eval_params = {'batch_size': FLAGS.batch_size}
    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                          args=eval_params)
    print '==> Accuracy : {}'.format(accuracy)

    def evaluate():
        # Evaluate the accuracy of the CIFAR10 model on legitimate test examples
        eval_params = {'batch_size': FLAGS.batch_size}
        accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                              args=eval_params)
        assert X_test.shape[0] == 10000, X_test.shape
        print('Test accuracy on legitimate test examples: ' + str(accuracy))

    # Train an CIFAR10 model
    # train_params = {
    #     'nb_epochs': FLAGS.nb_epochs,
    #     'batch_size': FLAGS.batch_size,
    #     'learning_rate': FLAGS.learning_rate
    # }

    im_base = '/im_'
    model_name = args_model + '_p' + str(args_pool)
    if args_attack == 'fgsm' or args_attack == 'FGSM':

        result_dir = os.getcwd() + '/images/fgsm/'
        print "==> creating fgsm adversarial wrapper"

        pdb.set_trace()
        adv_x = fgsm_old(x, predictions, eps=0.05)

        print "==> sending to batch evaluator to finalize adversarial images"
        eval_params = {'batch_size': FLAGS.batch_size}
        X_train_adv, = batch_eval(sess, [x], [adv_x], [X_train], args=eval_params)

        i = 0
        if not os.path.exists(result_dir + model_name):
            os.makedirs(result_dir + model_name)
        print "==> saving images to {}".format(result_dir + model_name)
        for ad in X_train_adv:
            scipy.misc.imsave(result_dir + model_name + im_base + str(i) + '.png', ad)
            i += 1

        sess.close()

    """ JSMA """
    if args_attack == 'jsma' or args_attack == 'JSMA':

        result_dir = os.getcwd() + '/images/jsma/trial_single_adv'
        print('Crafting ' + str(FLAGS.source_samples) + ' * ' +
              str(FLAGS.nb_classes - 1) + ' adversarial examples')

        results = np.zeros((FLAGS.nb_classes, FLAGS.source_samples), dtype='i')

        # This array contains the fraction of perturbed features for each test set
        perturbations = np.zeros((FLAGS.nb_classes, FLAGS.source_samples),
                                 dtype='f')

        # Define the TF graph for the model's Jacobian
        grads = jacobian_graph(predictions, x, FLAGS.nb_classes)

        # Initialize our array for grid visualization
        grid_shape = (FLAGS.nb_classes,
                      FLAGS.nb_classes,
                      FLAGS.img_rows,
                      FLAGS.img_cols,
                      FLAGS.nb_channels)
        grid_viz_data = np.zeros(grid_shape, dtype='f')
        i_saved = 0
        n_image = 0
        # Loop over the samples we want to perturb into adversarial examples
        print "==> saving images to {}".format(result_dir + model_name)
        for sample_ind in xrange(7166, FLAGS.source_samples):
            # We want to find an adversarial example for each possible target class
            current_class = int(np.argmax(Y_train[sample_ind]))
            target_classes = other_classes(FLAGS.nb_classes, current_class)
            # For the grid visualization, keep original images along the diagonal
            grid_viz_data[current_class, current_class, :, :, :] = np.reshape(
                X_train[sample_ind:(sample_ind + 1)],
                (FLAGS.img_rows, FLAGS.img_cols, FLAGS.nb_channels))

            # Loop over all target classes
            adversarials = []
            for idx, target in enumerate(target_classes):
                print "image {}".format(sample_ind)

                # here we hold all successful adversarials for this iteration
                # since we dont want 500k images, we will uniformly sample an image to save after each target

                print('--------------------------------------')
                print('Creating adv. example for target class ' + str(target))

                # This call runs the Jacobian-based saliency map approach
                adv_x, res, percent_perturb = SaliencyMapMethod(sess, x, predictions, grads,
                                                                X_train[sample_ind:
                                                                        (sample_ind + 1)],
                                                                target, theta=1, gamma=0.1,
                                                                increase=True, back='tf',
                                                                clip_min=0, clip_max=1)
                # Display the original and adversarial images side-by-side
                adversarial = np.reshape(adv_x, (FLAGS.img_rows, FLAGS.img_cols, FLAGS.nb_channels))
                original = np.reshape(X_train[sample_ind:(sample_ind + 1)],
                                      (FLAGS.img_rows, FLAGS.img_cols, FLAGS.nb_channels))

                if FLAGS.viz_enabled:

                    if 'figure' not in vars():
                        figure = pair_visual(original, adversarial)
                    else:
                        figure = pair_visual(original, adversarial, figure)

                if not os.path.exists(result_dir + model_name):
                    os.makedirs(result_dir + model_name)

                if res == 1:
                    adversarials.append(adversarial)

                if idx == FLAGS.nb_classes - 2:

                    try:
                        if len(adversarials) == 1:
                            idx_uniform = 0
                        else:
                            idx_uniform = np.random.randint(0, len(adversarials) - 1)
                        print idx_uniform
                        scipy.misc.imsave(result_dir + model_name + im_base +
                                          str(sample_ind) + '.png', adversarials[idx_uniform])
                        i_saved += 1
                        print "==> images saved: {}".format(i_saved)

                    except:

                        print "No adversarials generated"

                # Add our adversarial example to our grid data
                grid_viz_data[target, current_class, :, :, :] = np.reshape(
                    adv_x, (FLAGS.img_rows, FLAGS.img_cols, FLAGS.nb_channels))

                # Update the arrays for later analysis
                results[target, sample_ind] = res
                perturbations[target, sample_ind] = percent_perturb

            n_image += 1
        # Compute the number of adversarial examples that were successfuly found
        nb_targets_tried = ((FLAGS.nb_classes - 1) * FLAGS.source_samples)
        succ_rate = float(np.sum(results)) / nb_targets_tried
        print('Avg. rate of successful adv. examples {0:.2f}'.format(succ_rate))

        # Compute the average distortion introduced by the algorithm
        percent_perturbed = np.mean(perturbations)
        print('Avg. rate of perturbed features {0:.2f}'.format(percent_perturbed))

        # Compute the average distortion introduced for successful samples only
        percent_perturb_succ = np.mean(perturbations * (results == 1))
        print('Avg. rate of perturbed features for successful '
              'adversarial examples {0:.2f}'.format(percent_perturb_succ))

        # Close TF session
        sess.close()

        # Finally, block & display a grid of all the adversarial examples
        if FLAGS.viz_enabled:
            grid_visual(grid_viz_data)


if __name__ == '__main__':

    generate_images()