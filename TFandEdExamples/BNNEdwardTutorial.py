# from this link: https://www.alpha-i.co/blog/MNIST-for-ML-beginners-The-Bayesian-Way.html

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

x = tf.placeholder(tf.float32, [None, D])

# Normal(0, 1) priors for the variables. note that the syntax assumes Tensorflow 1.1
w = Normal(loc=tf.zeros([D,K]), scale=tf.ones(K))
b = Normal(loc=tf.zeros(K), scale=tf.ones(K))

y = Categorical(tf.matmul(x, w) + b)

qw = Normal(loc=tf.Variable(tf.random_normal([D, K])),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, K]))))
qb = Normal(loc=tf.Variable(tf.random_normal([K])),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))

y_ph = tf.placeholder(tf.int32, [N])

inference = ed.KLqp({w: qw,b: qb}, data={y: y_ph})

inference.initialize(n_iter=5000, n_print=100, scale={y: float(mnist.train.num_examples) / N})

# use interactive session
sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
    X_batch, Y_batch = mnist.train.next_batch(N)

    Y_batch = np.argmax(Y_batch, axis=1)
    info_dict = inference.update(feed_dict={x:X_batch, y_ph: Y_batch})
    inference.print_progress(info_dict)

# testing that model

#load data
X_test = mnist.test.images
Y_test = np.argmax(mnist.test.labels, axis=1)

print('hello')
n_samples = 100
prob_lst = []
samples = []
w_samples = []
b_samples = []

for _ in range(n_samples):
    w_samp = qw.sample()
    b_samp = qb.sample()
    w_samples.append(w_samp)
    b_samples.append(b_samp)

    prob = tf.nn.softmax(tf.matmul(X_test, w_samp) + b_samp)
    prob_lst.append(prob.eval())
    sample = tf.concat([tf.reshape(w_samp, [-1]), b_samp], 0)
    samples.append(sample.eval())

#compute accuracy of the model
accy_test = []
for prob in prob_lst:
    y_trn_prd = np.argmax(prob, axis=1).astype(np.float32)
    acc = (y_trn_prd == Y_test).mean()*100
    acc_test.append(acc)

plt.hist(accy_test)
plt.title("Histogram of prediction accuracies in the MNIST test data")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")


# compute mean probabilities for each class for all (W, b)

Y_pred = np.argmax(np.mean(prob_lst, axis=0), axis=1)
print("accuracy in predicting the test data = ", (Y_pred == Y_test).mean()*100)

#create a pandas dataframe of posterior samples
samples_df = pd.DataFrame(data = samples, index=range(n_samples))

samples_5 = pd.DataFrame(data = samples_df[list(range(5))].values, columns=["W_0", "W_1", "W_2", "W_3"]
