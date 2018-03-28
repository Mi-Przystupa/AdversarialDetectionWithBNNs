#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from edward.models import Categorical, Normal
import edward as ed
import pandas as pd

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

ed.set_seed(314159)
N = 100 
D = 784
K = 10

x = tf.placeholder(tf.float32, [None, 28, 28, 1])

#Convolution layer 1 

# 3x3 conv, pad w/ 0s on edges
# feats to compute on window
num_filters = 32 

conv1 = Normal(loc=tf.zeros([3, 3, 1, num_filters]), scale=tf.ones([3, 3, 1, num_filters]))
conv_b1 = Normal(loc=tf.zeros(num_filters), scale=tf.ones(num_filters))

conv2 = Normal(loc=tf.zeros([3, 3, num_filters, num_filters]), scale=tf.ones([3, 3, 1, num_filters]))
conv_b2 = Normal(loc=tf.zeros(num_filters), scale=tf.ones(num_filters))

#weight shape should match window size
#W1 = tf.Variable(tf.truncated_normal([winx, winy, 1, num_filters], \
#        stddev=1./math.sqrt(winx*winy)))
#b1 = tf.Variable(tf.constant(0.1, shape=[num_filters]))


#Dense layer
# Normal(0, 1) priors for the variables. note that the syntax assumes Tensorflow 1.1
w = Normal(loc=tf.zeros([(num_filters*49),K]), scale=tf.ones(K))
b = Normal(loc=tf.zeros(K), scale=tf.ones(K))


#Building graph
xw = tf.nn.conv2d(x, conv1, strides=[1,2,2,1], padding='SAME')
h1 = tf.nn.relu(xw + conv_b1)

h2 = tf.nn.conv2d(h1, conv2, strides=[1,2,2,1], padding='SAME')
h2 = tf.nn.relu(h2 + conv_b2)
h2 = tf.reshape(h2, [-1, num_filters*49])

y = Categorical(tf.matmul(h2, w) + b)

# the Q distribution

qconv1 = Normal(loc=tf.Variable(tf.random_normal([3, 3, 1, num_filters])),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([3, 3, 1, num_filters]))))
qconv_b1 = Normal(loc=tf.Variable(tf.random_normal([num_filters])),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([num_filters]))))

qconv2 = Normal(loc=tf.Variable(tf.random_normal([3, 3, num_filters, num_filters])),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([3, 3, num_filters, num_filters]))))
qconv_b2 = Normal(loc=tf.Variable(tf.random_normal([num_filters])),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([num_filters]))))


qw = Normal(loc=tf.Variable(tf.random_normal([num_filters* (D / 16), K])),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([num_filters* (D / 16), K]))))
qb = Normal(loc=tf.Variable(tf.random_normal([K])),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))

y_ph = tf.placeholder(tf.int32, [N])

inference = ed.KLqp({conv1: qconv1, conv_b1: qconv_b1,conv2: qconv2, conv_b2: qconv_b2, w: qw,b: qb}, data={y: y_ph})

inference.initialize(n_iter=5000, n_print=100, scale={y: float(mnist.train.num_examples) / N})

# use interactive session
sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
    X_batch, Y_batch = mnist.train.next_batch(N)
    X_batch = np.reshape(X_batch, (-1, 28, 28 , 1))
    Y_batch = np.argmax(Y_batch, axis=1)
    info_dict = inference.update(feed_dict={x:X_batch, y_ph: Y_batch})
    inference.print_progress(info_dict)


