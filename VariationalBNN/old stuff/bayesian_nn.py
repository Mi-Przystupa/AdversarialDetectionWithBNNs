"""
Bayesian neural network using variational inference
(see, e.g., Blundell et al. (2015); Kucukelbir et al. (2016)).

Inspired by autograd's Bayesian neural network example.
This example prettifies some of the tensor naming for visualization in
TensorBoard. To view TensorBoard, run `tensorboard --logdir=log`.

References
----------
http://edwardlib.org/tutorials/bayesian-neural-network
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.examples.tutorials.mnist import input_data
from edward.models import Categorical, Normal,OneHotCategorical
import edward as ed
import pandas as pd
import glob
import Uncertainty



from edward.models import Normal

tf.flags.DEFINE_integer("N", default_value=55000, docstring="Number of data points.")
tf.flags.DEFINE_integer("D", default_value=784, docstring="Number of features.")
tf.flags.DEFINE_integer("outputs", default_value=10, docstring="number of outputs")

FLAGS = tf.flags.FLAGS


def GetMNISTDataset():
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    return mnist

def Evaluation(network, X_test, Y_test, X,  n_samples=20, plot=False, title='Historgram of Accuracies'):
    print('let evaluate')
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run() 
    prob_lst = [] 
    for _ in range(0, n_samples): 
        h = X_test
        print(network.sample())
        output = network.sample().eval(feed_dict={X: X_test})
        prob = output
        prob_lst.append(prob)
    if plot:
        PlotSamplesHistogram(prob_lst, Y_test, title)

    return prob_lst 

def PlotSamplesHistogram(probs, Y_test, title):
    acc_test =[]
    for prob in probs:
        y_trn_prd = np.argmax(prob, axis=1).astype(np.float32)
        acc = (y_trn_prd == Y_test).mean()* 100
        print(acc)
        acc_test.append(acc)

    plt.figure()
    plt.hist(acc_test)
    plt.title("Histogram of prediction accuracies in the MNIST test data")
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()



HIDDEN_SIZE =  50 
def main(_):
  def neural_network(X):
    h = tf.tanh(tf.matmul(X, W_0) + b_0)
    h = tf.tanh(tf.matmul(h, W_1) + b_1)
    h = tf.matmul(h, W_2) + b_2
    return tf.nn.softmax(tf.reshape(h, [-1, FLAGS.outputs]))
  ed.set_seed(42)

  # DATA
  mnist = GetMNISTDataset()
  train = mnist.train 
  X_train, y_train = train.next_batch(len(train._images))
  y_train = y_train.astype(np.int)
  print(type(y_train))
  print(X_train.shape)
  print(y_train.shape)
  # MODEL
  with tf.name_scope("model"):
    W_0 = Normal(loc=tf.zeros([FLAGS.D, HIDDEN_SIZE]), scale=tf.ones([FLAGS.D, HIDDEN_SIZE]),
                 name="W_0")
    W_1 = Normal(loc=tf.zeros([HIDDEN_SIZE, HIDDEN_SIZE]), scale=tf.ones([HIDDEN_SIZE, HIDDEN_SIZE]), name="W_1")
    W_2 = Normal(loc=tf.zeros([HIDDEN_SIZE, FLAGS.outputs]), scale=tf.ones([HIDDEN_SIZE, FLAGS.outputs]), name="W_2")
    b_0 = Normal(loc=tf.zeros(HIDDEN_SIZE), scale=tf.ones(HIDDEN_SIZE), name="b_0")
    b_1 = Normal(loc=tf.zeros(HIDDEN_SIZE), scale=tf.ones(HIDDEN_SIZE), name="b_1")
    b_2 = Normal(loc=tf.zeros(FLAGS.outputs), scale=tf.ones(FLAGS.outputs), name="b_2")

    X = tf.placeholder(tf.float32, [None, FLAGS.D], name="X")
    y = OneHotCategorical(probs=neural_network(X),name="y")

  # INFERENCE
  with tf.variable_scope("posterior"):
    with tf.variable_scope("qW_0"):
      loc = tf.get_variable("loc", [FLAGS.D, HIDDEN_SIZE], initializer=tf.random_normal_initializer())
      scale = tf.nn.softplus(tf.get_variable("scale", [FLAGS.D, HIDDEN_SIZE]))
      qW_0 = Normal(loc=loc, scale=scale, name='LOVEME')
    with tf.variable_scope("qW_1"):
      loc = tf.get_variable("loc", [HIDDEN_SIZE, HIDDEN_SIZE], initializer=tf.random_normal_initializer())
      scale = tf.nn.softplus(tf.get_variable("scale", [HIDDEN_SIZE, HIDDEN_SIZE]))
      qW_1 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qW_2"):
      loc = tf.get_variable("loc", [HIDDEN_SIZE, FLAGS.outputs], initializer=tf.random_normal_initializer())
      scale = tf.nn.softplus(tf.get_variable("scale", [HIDDEN_SIZE, FLAGS.outputs]))
      qW_2 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_0"):
      loc = tf.get_variable("loc", [HIDDEN_SIZE], initializer=tf.random_normal_initializer())
      scale = tf.nn.softplus(tf.get_variable("scale", [HIDDEN_SIZE]))
      qb_0 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_1"):
      loc = tf.get_variable("loc", [HIDDEN_SIZE], initializer=tf.random_normal_initializer())
      scale = tf.nn.softplus(tf.get_variable("scale", [HIDDEN_SIZE]))
      qb_1 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_2"):
      loc = tf.get_variable("loc", [FLAGS.outputs], initializer=tf.random_normal_initializer())
      scale = tf.nn.softplus(tf.get_variable("scale", [FLAGS.outputs]))
      qb_2 = Normal(loc=loc, scale=scale)

  inference = ed.KLqp( {W_0: qW_0, b_0: qb_0,
                       W_1: qW_1, b_1: qb_1,
                       W_2: qW_2, b_2: qb_2}, data={X: X_train, y: y_train})
  optimizer = None#tf.train.AdamOptimizer(learning_rate=0.00001) 
  inference.run(optimizer=optimizer,n_samples=1, n_iter=2000,logdir='log')
  y_post = ed.copy(y, {W_0: qW_0, b_0: qb_0,
                       W_1: qW_1, b_1: qb_1,
                       W_2: qW_2, b_2: qb_2})
  #print('we got here')
  X_test =  mnist.test.images
  Y_test = np.argmax(mnist.test.labels,axis=1)
  #Y_test = mnist.test.labels

  Evaluation(y_post, X_test, Y_test,X,  n_samples=50, plot=True, title='Historgram of Accuracies')



if __name__ == "__main__":
  tf.app.run()

