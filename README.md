# AdversarialDetectionWithBNNs


##Variational Inference
Used Theano implementation from : https://github.com/HIPS/Probabilistic-Backpropagation

## Adversaries we have so far:

We have FGSM epsilon values in the range [0.01, 0.03, 0.07, 0.1, 0.2, 0.3].

FGSM on MNIST: http://www.cs.ubc.ca/~cfung1/adv_images/mnist_adv_examples.zip  
JSMA on MNIST: http://www.cs.ubc.ca/~cfung1/adv_images/jsma_adv_examples_10000.zip  

FGSM on CIFAR-10: http://www.cs.ubc.ca/~cfung1/fgsm/vgg_fgsm_0.1/im_0.png  
This is just a pattern with the link at vgg_fgsm_EPSILON and im_INDEX.png  

CIFAR-10 labels: http://www.cs.ubc.ca/~cfung1/fgsm/vgg_adv_y_10000.npy  

Gaussian Noise: http://www.cs.ubc.ca/~cfung1/gaussian/gaussian_examples.zip  
