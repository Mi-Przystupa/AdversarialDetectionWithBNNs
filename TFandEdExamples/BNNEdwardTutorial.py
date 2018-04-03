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
n_samples = 20
prob_lst = []
samples = []
w_samples = []
b_samples = []

for _ in range(n_samples):
    print('hello again: {}'.format(_))
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
    accy_test.append(acc)

print(accy_test)
plt.show(plt.hist(accy_test))
plt.title("Histogram of prediction accuracies in the MNIST test data")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")



# compute mean probabilities for each class for all (W, b)

Y_pred = np.argmax(np.mean(prob_lst, axis=0), axis=1)
print("accuracy in predicting the test data = ", (Y_pred == Y_test).mean()*100)

#create a pandas dataframe of posterior samples
samples_df = pd.DataFrame(data = samples, index=range(n_samples))

samples_5 = pd.DataFrame(data = samples_df[list(range(5))].values, columns=["W_0", "W_1", "W_2", "W_3", "W_4"])

#use seaborn pairgrid to make triale plot to show auto & cross correlations

g = sns.PairGrid(samples_5, diag_sharey=False)
g.map_lower(sns.kdeplot, n_levels = 4, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot,legend=False)
plt.subplots_adjust(top=0.95)
plt.subplots_adjust(top=0.95)

g.fig.suptitle('Joint posterior distribution of the first 5 weights')
plt.show(g)

#Load the first image from the test data and its label 

test_image = X_test[0:1]
test_label = Y_test[0]
print('truth = ', test_label)
pixels = test_image.reshape((28,28))
plt.imshow(pixels,cmap='Blues')

#Now check what the model predicts for each (w,b) sample from posterior

sing_img_probs = []

for w_samp, b_samp in zip(w_samples, b_samples):
    prob = tf.nn.softmax(tf.maxmul(X_test[0:1], w_samp) + b_samp)
    sing_img_probs.append(prob.eval())

hist = plt.hist(np.argmax(sing_img_probs, axis=2), bins=range(10))
plt.xticks(np.arange(0, 10))
plt.xlim(0, 10)
plt.xlabel("Accuracy of the prediction of the test digit")
plt.ylabel("Frequency")

plt.show(hist)


# un familiar data
