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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from edward.models import Categorical, Normal
import edward as ed
import pandas as pd
import glob

N = 100
D = 784

H1 = 392
H2 = 146
H3 =  73
K = 10

adversaries = glob.glob('./examples/*_x_*')
labels = glob.glob('./examples/*_y_*')

layers = [D, K] #H1, H2, H3, K]

def GetMNISTDataset():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 
    return mnist

def BuildModelDynamic(x):

    pq = {}
    q = []
    for i, value in enumerate(layers):
        if( i == len(layers) -1):
            break
        inputs = value
        outputs = layers[i+1]
        w = Normal(tf.zeros([inputs, outputs]), scale=tf.ones(outputs))
        b = Normal(tf.zeros(outputs), scale=tf.ones(outputs))
        x = tf.nn.tanh(tf.matmul(x, w) + b)

        qw = Normal(loc=tf.get_variable('loc/qw_'+str(i), [inputs,outputs]),
                scale=tf.get_variable('scale/qw_' + str(i), [inputs, outputs]))
        qb = Normal(loc=tf.get_variable('loc/qb_' + str(i), [outputs]),
                scale=tf.nn.softplus(tf.get_variable('scale/qb_'+str(i), [outputs])))
        
        pq[w] = qw
        pq[b] = qb
        q.append({'qw':qw,'qb': qb})
    y = Categorical(x)
    y_ph = tf.placeholder(tf.int32, [N]) 
    inference = ed.KLqp(pq, data={y: y_ph})
    return inference, y_ph, y, q 


def Train(inference, dataset, x, y_ph):
    # use interactive session
    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    variables = [v for v in tf.global_variables()]

    for _ in range(inference.n_iter):
        X_batch, Y_batch = dataset.train.next_batch(N)

        Y_batch = np.argmax(Y_batch, axis=1)
        info_dict = inference.update(feed_dict={x:X_batch, y_ph: Y_batch})
        inference.print_progress(info_dict)



def Evaluation(layers, X_test, Y_test,  n_samples=20, plot=False, title='Historgram of Accuracies'):
    
    prob_lst = [] 
    for _ in range(0, n_samples): 
        h = X_test
        for l in  layers:
            w_sample = l['qw'].sample()
            b_sample = l['qb'].sample()
            h = tf.matmul(h, w_sample) + b_sample  
        prob = tf.nn.softmax(h)
        prob_lst.append(prob.eval())
    if plot:
        PlotResults(prob_lst, Y_test, title)

    return prob_lst 

def PlotResults(probs, Y_test, title):

    acc_test =[]
    for prob in probs:
        y_trn_prd = np.argmax(prob, axis=1).astype(np.float32)
        acc = (y_trn_prd == Y_test).mean()* 100
        acc_test.append(acc)

    plt.figure()
    plt.hist(acc_test)
    plt.title("Histogram of prediction accuracies in the MNIST test data")
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()

def main():
    #Get the data
    mnist = GetMNISTDataset()
    x = tf.placeholder(tf.float32,[None, D], name='input')

    
    inference, y_ph, y, q  = BuildModelDynamic(x) 
    variables = [v for v in tf.global_variables()]
    for v in variables:
        print(v.name)
    inference.initialize(n_iter=5000, n_print=100, scale={y: float(mnist.train.num_examples) / N})
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()


    X_test = mnist.test.images
    Y_test = np.argmax(mnist.test.labels,axis=1)

    #train the model
    Train(inference, mnist, x, y_ph)

    #measure it's perform
    #probs = Evaluation(q,X_test, Y_test, plot=True) 

    Y_test = np.load(labels[0])
    Y_test = np.argmax(Y_test,axis=1)
    for a in adversaries:
        X_test = np.load(a)
        X_test = np.reshape(X_test, (-1, D))
        probs = Evaluation(q,X_test, Y_test, plot=True, title=a)

if __name__ == "__main__":
    main()

