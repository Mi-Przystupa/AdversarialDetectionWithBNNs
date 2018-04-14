from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import logging
import pdb

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from edward.models import Categorical, Normal
import edward as ed
import pandas as pd
import glob
import Uncertainty

N = 100
D = 784

H1 = 392
H2 = 146
H3 =  73
K = 10

adversaries = glob.glob('./examples/*_x_*')
print(adversaries)
labels = glob.glob('./examples/*_y_*')

layers = [D, H1, H2, H3, K]

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
        if(i == (len(layers) - 1)):
            x = tf.nn.softmax(tf.matmul(x, w) + b)
        else:
            x = tf.nn.relu(tf.matmul(x, w) + b)

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
    inference.finalize()



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

def PlotSamplesHistogram(probs, Y_test, title):
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
    inference.initialize(n_iter=5000, n_print=100, scale={y: float(mnist.train.num_examples) / N},logdir='log' )
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()


    X_test = mnist.test.images
    Y_test = np.argmax(mnist.test.labels,axis=1)

    #train the model
    Train(inference, mnist, x, y_ph)

    #measure it's perform
    probs = Evaluation(q,X_test, Y_test, plot=True) 

    #Y_test = np.load(labels[0])
    Y_test = np.argmax(Y_test,axis=1)
    for a in adversaries:
        X_test = np.load(a)
        X_test = np.reshape(X_test, (-1, D))
        probs = Evaluation(q,X_test, Y_test, plot=False, title=a)

        print(a)
        variationRatio = []
        mutualInformation = []
        predictiveEntropy = []
        predictions = []
        probs = np.array(probs)

        for i in range(0, len(Y_test)):
            p = probs[:,i, :] 
            variationRatio.append(Uncertainty.variation_ratio(p))
            mutualInformation.append(Uncertainty.mutual_information(p))
            predictiveEntropy.append(Uncertainty.predictive_entropy(p))
            predictions.append(np.max(p.mean(axis=1), axis=0))

            
        uncertainty={}
        uncertainty['varation_ratio']= np.array(variationRatio)
        uncertainty['predictive_entropy']= np.array(predictiveEntropy)
        uncertainty['mutual_information']= np.array(mutualInformation)
        predictions = np.array(predictions)

        a = a.split('_')[-1]
        a = a.split('.npy')[0]
        a = float(a)
        if(0.04999999 <= a and a <= 0.050000001):
            plot_uncertainty(uncertainty,predictions,adversarial_type='fgsm',epsilon=a)
        elif (0.2999999 <= a and a <= .300000001):
            plot_uncertainty(uncertainty,predictions,adversarial_type='fgsm',epsilon=a)


        print(a)
        
if __name__ == "__main__":
    main()

